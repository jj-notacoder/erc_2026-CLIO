"""Actual ROS-class method wiring, with fake transports and bounded fit spies.

No node/executor, robot commands, solver or official mesh load is started.
The registered target and obstacle checks execute their real NumPy geometry.
"""
from collections import deque
import copy
import itertools
import json
import math
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip('rclpy')
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import TransformException

from erc_phase1_solution import manipulation_node as manipulation
from erc_phase1_solution import perception_node as perception
from erc_phase1_solution import scene_checked_place as geometry
from erc_phase1_solution.bin_geometry import BinTracker
from erc_phase1_solution.bin_scene_admission import (
    BIN_MODEL, BIN_MESH_SHA256, CAD_RADIUS, CAVITY_BOUNDS, OUTER_BOUNDS,
)
from erc_phase1_solution.mission_manager import MissionManager


class Clock:
    def __init__(self, ns=1_200_000_000): self.ns=ns
    def now(self): return SimpleNamespace(nanoseconds=self.ns)
    def monotonic(self): return self.ns/1e9
    def sleep(self, seconds): self.ns += round(seconds*1e9)


def fitted_scene():
    yaw=.37
    rotation=np.array([[-math.sin(yaw),0.,math.cos(yaw)],
                       [math.cos(yaw),0.,math.sin(yaw)],[0.,1.,0.]])
    origin=np.array([.982,.044,.845])
    return dict(valid=True, model=BIN_MODEL, frame='base_footprint', mesh_sha256=BIN_MESH_SHA256,
        origin=origin.tolist(), rotation=rotation.tolist(),
        floor_center=(origin+rotation@[0.,-.095,0.]).tolist(), floor_point_model=[0.,-.095,0.],
        outer_bounds=[list(x) for x in OUTER_BOUNDS], cavity_bounds=[list(x) for x in CAVITY_BOUNDS],
        modeled_margin_m=.005, translation_error_m=.006, rotation_error_rad=.01,
        registration_margin_m=.006+2*CAD_RADIUS*math.sin(.005),
        quality=dict(floor_rear_registration_observed=True,accepted_candidates=1,
            finite_cad_coverage=.98,finite_surface_residual_p95_m=.001,
            feature_indices=[0,1,2]))


def epoch_entry(stamp=1_000_000_000):
    return dict(mode='bin',event='bin_verified',bin_valid=True,observation_stamp_ns=stamp,
        bin_floor_point_base=[.89,.007,.751],
        table_scene=dict(valid=True,frame='base_footprint',origin=[.98,.04,.74],
                         rotation=np.eye(3).tolist(),tag='same-depth-table'),
        bin_scene=fitted_scene(),scene_base_reference=dict(frame_id='odom',child_frame_id='base_footprint',
            requested_stamp_ns=stamp,producer_stamp_ns=stamp,pose=[.1,-.2,.3]))


def stamped_point(stamp=1_000_000_000, xyz=(.89,.007,.751)):
    point=PointStamped();point.header.frame_id='base_footprint'
    point.header.stamp.sec,point.header.stamp.nanosec=divmod(stamp,1_000_000_000)
    point.point.x,point.point.y,point.point.z=map(float,xyz)
    return point


def consumer(monkeypatch, *, required=True):
    node=object.__new__(manipulation.ManipulationNode)
    clock=Clock()
    parked=[*[f'arm_right_{i}_joint' for i in range(1,8)],'head_1_joint','head_2_joint']
    node._lock=threading.RLock();node._cancel=threading.Event()
    node.joints=dict.fromkeys(parked,0.);node._joint_stamps_ns=dict.fromkeys(parked,clock.ns)
    node.right_chain=SimpleNamespace(active_names=('torso_lift_joint',*parked[:7]))
    node.head_chain=SimpleNamespace(active_names=('torso_lift_joint',*parked[7:]))
    node._staging_odom=dict(stamp_ns=clock.ns,producer_stamp_ns=clock.ns,
        frame_id='odom',child_frame_id='base_footprint',pose=[.1,-.2,.3],linear_speed=0.,angular_speed=0.)
    node.get_clock=lambda:clock
    node.table_scene_required=required;node.bin_scene_required=required
    node.bin_candidate=None;node.latest_bin=None;node.bin_verified_ns=-1;node.bin_invalidated_ns=-1
    node.perception_wait=.11
    node._point_in_base=lambda message:np.array([message.point.x,message.point.y,message.point.z])
    monkeypatch.setattr(manipulation,'time',clock)
    return node,clock


def deliver(node, entry=None, point=None, *, status_first=False):
    entry=epoch_entry() if entry is None else entry
    point=stamped_point() if point is None else point
    status=String(data=json.dumps(entry))
    if status_first: node._on_bin_status(status)
    node._on_bin(point)
    if not status_first: node._on_bin_status(status)
    return point


def perception_fixture(monkeypatch, *, enabled=True, failed_fit=False, missing_epoch=False):
    path=Path(perception.__file__).resolve().parents[1]/'test/data/bin_geometry/real_bin_rgbd.npz'
    with np.load(path) as data:
        bgr=data['bgr'].copy(); depths=data['depth'].copy(); k=data['k'].copy(); matrix=data['base_from_camera'].copy()
    # Exercise actual CvBridge conversion RGB8 -> BGR8 and unsigned millimetres.
    millimetres=np.rint(np.nan_to_num(depths,nan=0.,posinf=0.,neginf=0.)*1000).astype(np.uint16)
    node=object.__new__(perception.PerceptionNode)
    clock=Clock(10_100_000_000);node.get_clock=lambda:clock;node.mode='bin';node.bridge=CvBridge()
    node.bin_rgb_frames=deque();node.bin_depth_frames=deque();node.camera_info=SimpleNamespace(k=k)
    node.bin_table_height=.73;node.last_bin_consumed_ns=-1;node.last_bin_depth_ns=-1
    node.last_bin_verified_ns=-1;node.last_bin_observed_ns=-1;node.bin_tracker=BinTracker()
    node.bin_pub=Mock();node.status_pub=Mock();node.saved_modes={'bin'}
    node.table_scene_enabled=enabled;node.bin_scene_enabled=enabled
    calls=[]
    table=dict(valid=True,frame='base_footprint',origin=[.98,.04,.74],tag='selected-table')
    table_fit=Mock(return_value=table)
    def fit(rgb, depth, intrinsics, base_from_camera, detection, *, table_scene):
        calls.append((rgb.copy(),depth.copy(),np.array(intrinsics),base_from_camera.copy(),detection,table_scene))
        return dict(valid=False,reason='insufficient_planes',frame='base_footprint') if failed_fit else fitted_scene()
    fit_spy=Mock(side_effect=fit)
    monkeypatch.setattr(perception,'fit_table_scene',table_fit)
    monkeypatch.setattr(perception,'fit_bin_scene',fit_spy)
    transforms=[]
    def lookup(target,source,stamp):
        transforms.append((target,source,stamp.nanoseconds))
        if target=='odom' and missing_epoch: raise TransformException('epoch unavailable')
        result=TransformStamped();result.header.frame_id=target;result.child_frame_id=source
        result.header.stamp.sec,result.header.stamp.nanosec=divmod(stamp.nanoseconds,1_000_000_000)
        result.transform.rotation.w=1.
        if target=='odom':
            result.transform.translation.x=.1;result.transform.translation.y=-.2
        return result
    node.tf_buffer=SimpleNamespace(lookup_transform=lookup)
    def convert(translation,rotation):
        if translation.x==.1:
            result=np.eye(4);result[:2,3]=[.1,-.2];return result
        return matrix.copy()
    monkeypatch.setattr(perception,'camera_transform',convert)
    for rgb_ns in (10_000_000_000,10_100_000_000,10_200_000_000):
        rgb=node.bridge.cv2_to_imgmsg(bgr[:,:,::-1].copy(),encoding='rgb8')
        rgb.header.frame_id='rgb_optical_frame';rgb.header.stamp.sec,rgb.header.stamp.nanosec=divmod(rgb_ns,1_000_000_000)
        depth=node.bridge.cv2_to_imgmsg(millimetres,encoding='16UC1')
        depth.header.frame_id='depth_optical_frame';depth.header.stamp.sec,depth.header.stamp.nanosec=divmod(rgb_ns+20_000_000,1_000_000_000)
        clock.ns=rgb_ns+100_000_000
        node._on_rgb(rgb);node._on_depth(depth);node._process_bin()
    statuses=[json.loads(call.args[0].data) for call in node.status_pub.publish.call_args_list]
    verified=next(x for x in statuses if x['event']=='bin_verified')
    return SimpleNamespace(node=node,clock=clock,bgr=bgr,millimetres=millimetres,k=k,matrix=matrix,
        fit_calls=calls,fit_spy=fit_spy,table_fit=table_fit,table=table,transforms=transforms,verified=verified)


def test_actual_rgbd_callbacks_pass_bgr_metric_depth_and_exact_depth_epoch(monkeypatch):
    f=perception_fixture(monkeypatch)
    assert len(f.fit_calls)==1
    rgb,depth,k,matrix,_,table=f.fit_calls[0]
    np.testing.assert_array_equal(rgb,f.bgr)
    np.testing.assert_allclose(depth,f.millimetres.astype(float)/1000,rtol=0,atol=1e-6)
    assert depth.dtype.kind=='f' and f.node.latest_depth.dtype==np.uint16
    np.testing.assert_array_equal(k,f.k);np.testing.assert_array_equal(matrix,f.matrix)
    assert table is f.table
    assert f.transforms[-2:]==[('base_footprint','depth_optical_frame',10_220_000_000),('odom','base_footprint',10_220_000_000)]
    reference=f.verified['scene_base_reference']
    assert reference['requested_stamp_ns']==reference['producer_stamp_ns']==10_220_000_000
    point=f.node.bin_pub.publish.call_args.args[0]
    assert point.header.frame_id=='depth_optical_frame' and point.header.stamp.nanosec==220_000_000
    np.testing.assert_allclose(f.matrix[:3,:3]@[point.point.x,point.point.y,point.point.z]+f.matrix[:3,3],f.verified['bin_floor_point_base'])
    assert np.linalg.norm(np.asarray(f.verified['bin_floor_point_base'])-f.verified['bin_scene']['floor_center'])>.01


def test_failed_fit_preserves_actual_mission_navigation_but_required_place_rejects(monkeypatch):
    f=perception_fixture(monkeypatch,failed_fit=True)
    assert f.verified['bin_valid'] is True and f.verified['bin_scene']['valid'] is False
    point=f.node.bin_pub.publish.call_args.args[0]
    mission=object.__new__(MissionManager)
    mission.get_clock=lambda:f.clock;mission.bin_point=None;mission.bin_invalidated_ns=-1;mission.ready={}
    mission._on_bin(point);mission._on_perception_status(String(data=json.dumps(f.verified)))
    assert mission.bin_point is point
    node,clock=consumer(monkeypatch);clock.ns=f.clock.ns
    node._joint_stamps_ns=dict.fromkeys(node.joints,clock.ns)
    node._staging_odom.update(stamp_ns=clock.ns,producer_stamp_ns=clock.ns)
    node._point_in_base=lambda message:np.array(f.verified['bin_floor_point_base'])
    deliver(node,f.verified,point)
    assert node.latest_bin is point
    with pytest.raises(RuntimeError,match='bin_unregistered'):
        node._wait_for_perception_point('latest_bin')
    assert not hasattr(node,'_selected_place_bin_scene')


def test_missing_depth_epoch_odom_transform_has_no_registered_pose_fallback(monkeypatch):
    f=perception_fixture(monkeypatch,missing_epoch=True)
    f.node.bin_pub.publish.assert_called_once()
    assert f.verified['bin_valid'] is True
    assert f.verified['bin_scene']['valid'] is False
    assert 'epoch unavailable' in f.verified['bin_scene']['reason']
    assert 'scene_base_reference' not in f.verified


def test_disabled_perception_keeps_legacy_point_and_old_tf_signature(monkeypatch):
    f=perception_fixture(monkeypatch,enabled=False)
    f.fit_spy.assert_not_called();f.table_fit.assert_not_called()
    assert all(target=='base_footprint' for target,_,_ in f.transforms)
    assert 'bin_scene' not in f.verified and 'scene_base_reference' not in f.verified
    assert f.verified['bin_valid'] is True
    f.node.bin_pub.publish.assert_called_once()


def test_actual_consumer_accepts_both_callback_orders_without_mutating_navigation_point(monkeypatch):
    for status_first in (False,True):
        node,_=consumer(monkeypatch);point=deliver(node,status_first=status_first)
        admitted=node._wait_for_perception_point('latest_bin')
        np.testing.assert_allclose(admitted,fitted_scene()['floor_center'])
        assert node.latest_bin is point
        np.testing.assert_array_equal([point.point.x,point.point.y,point.point.z],[.89,.007,.751])
        assert node._selected_place_table_scene['tag']=='same-depth-table'


def test_actual_consumer_rejects_mixed_table_or_transform_epochs(monkeypatch):
    for mutation in ('bundle_stamp','transform_stamp','table_frame'):
        node,_=consumer(monkeypatch);entry=epoch_entry()
        if mutation=='bundle_stamp':entry['observation_stamp_ns']+=1
        elif mutation=='transform_stamp':entry['scene_base_reference']['producer_stamp_ns']-=1
        else:entry['table_scene']['frame']='odom'
        deliver(node,entry)
        # Preserve current-point verification so this test reaches scene binding.
        node.bin_verified_ns=1_000_000_000;node.latest_bin=node.bin_candidate
        with pytest.raises(RuntimeError,match='placement_scene_'):
            node._wait_for_perception_point('latest_bin')
        assert not hasattr(node,'_selected_place_bin_scene')


def test_callback_replacement_during_admission_keeps_one_epoch_and_owned_scene_copies(monkeypatch):
    node,_=consumer(monkeypatch);deliver(node)
    original=node._latest_table_scene_entry
    real=manipulation.measured_scene_context
    def context(n):
        value=real(n)
        replacement=epoch_entry(1_010_000_000);replacement['bin_scene']['origin'][0]+=.2
        replacement['bin_scene']['floor_center'][0]+=.2
        n._on_bin_status(String(data=json.dumps(replacement)))
        return value
    monkeypatch.setattr(manipulation,'measured_scene_context',context)
    admitted=node._wait_for_perception_point('latest_bin')
    np.testing.assert_allclose(admitted,fitted_scene()['floor_center'])
    original['bin_scene']['quality']['feature_indices'][0]=99
    original['table_scene']['origin'][0]=99
    assert node._selected_place_bin_scene['quality']['feature_indices'][0]==0
    assert node._selected_place_table_scene['origin'][0]!=99
    assert node._latest_table_scene_entry['observation_stamp_ns']==1_010_000_000


def test_actual_odom_callback_preserves_raw_metadata_and_zero_stamp_is_not_repaired_for_scene(monkeypatch):
    node,clock=consumer(monkeypatch)
    message=Odometry();message.header.frame_id='odom';message.child_frame_id='base_footprint'
    message.header.stamp.sec=1;message.header.stamp.nanosec=190_000_000
    message.pose.pose.position.x=.1;message.pose.pose.position.y=-.2
    message.pose.pose.orientation.z=math.sin(.15);message.pose.pose.orientation.w=math.cos(.15)
    node._on_staging_odom(message)
    context=manipulation.measured_scene_context(node)
    assert context['odom_frame_id']=='odom' and context['odom_child_frame_id']=='base_footprint'
    assert context['odom_producer_stamp_ns']==1_190_000_000
    deliver(node);np.testing.assert_allclose(node._wait_for_perception_point('latest_bin'),fitted_scene()['floor_center'])
    message.header.stamp.sec=0;message.header.stamp.nanosec=0
    node._on_staging_odom(message)
    assert node._staging_odom['producer_stamp_ns']==0
    with pytest.raises(RuntimeError,match='odometry_stamp|stale_odometry'):
        node._wait_for_perception_point('latest_bin')


class PlanningObserved(RuntimeError): pass


def place_node(monkeypatch, *, registered):
    node,_=consumer(monkeypatch,required=registered)
    node._held_book_corners=np.array(list(itertools.product((-.08,.08),(-.01,.01),(-.125,.125))))+[.055,-.001,.013]
    node._carried_staging_solution=np.array([.35,.4,.5,.6,.7,.8,.9,1.])
    node._gravity_supported_payload=True;node.book_centered_place_enabled=True
    node.book_centered_place_height_above_point=.27;node.book_centered_place_approach_x=.60
    node.book_centered_place_clearance_height=.50;node.place_torso_height=.35;node.cartesian_step=.06
    node._adaptive_command_guard=lambda:node._lock
    node._fresh_retention_probe=lambda *a,**kw:True
    node._measured_left_solution=lambda:np.array([.35,1.,1.,1.,1.,1.,1.,1.])
    node._interpolate_positions=lambda a,b,step:[b.copy()]
    node._move_torso=Mock(side_effect=AssertionError('unexpected motion'))
    node._open_gripper=Mock(side_effect=AssertionError('unexpected release'))
    if registered:deliver(node)
    else:
        node._wait_for_perception_point=lambda key:np.array([.89,.007,.751])
    return node


def test_actual_place_forwards_registered_floor_rotation_table_and_bin_to_planner(monkeypatch):
    node=place_node(monkeypatch,registered=True);calls=[]
    def planner(n,positions,rotation,carried,torso,seed,point,resolver,*,table_scene=None,bin_scene=None):
        calls.append((positions,rotation,point,table_scene,bin_scene))
        raise PlanningObserved()
    monkeypatch.setattr(manipulation,'plan_scene_checked_place',planner)
    with pytest.raises(PlanningObserved):node._place()
    positions,rotation,point,table,scene=calls[0]
    np.testing.assert_allclose(point,fitted_scene()['floor_center'])
    assert table['tag']=='same-depth-table' and scene is node._selected_place_bin_scene
    placed=node._held_book_corners@rotation.T+positions[-1]
    local=(placed-np.asarray(scene['floor_center']))@np.asarray(scene['rotation'])
    np.testing.assert_allclose(local.mean(axis=0),[0,.27,0],atol=1e-12)
    np.testing.assert_allclose(local[4]-local[0],[-.16,0,0],atol=1e-12)
    np.testing.assert_allclose(local[1]-local[0],[0,0,.25],atol=1e-12)
    node._move_torso.assert_not_called();node._open_gripper.assert_not_called()


def test_legacy_place_still_calls_old_planner_signature_without_registered_keywords(monkeypatch):
    node=place_node(monkeypatch,registered=False);seen=[]
    # Deliberately no **kwargs or bin_scene parameter: legacy callers still work.
    def planner(n,positions,rotation,carried,torso,seed,point,resolver,*,table_scene=None):
        seen.append((point,table_scene));raise PlanningObserved()
    monkeypatch.setattr(manipulation,'plan_scene_checked_place',planner)
    with pytest.raises(PlanningObserved):node._place()
    np.testing.assert_array_equal(seen[0][0],[.89,.007,.751]);assert seen[0][1] is None


def test_real_scene_planner_rejects_missing_required_bin_before_any_mesh_load(monkeypatch):
    node=SimpleNamespace(_cancel=threading.Event(),bin_scene_required=True,gripper_open=.069,
        _adaptive_motion_feedback=lambda:(SimpleNamespace(position=.017),None))
    load=Mock(side_effect=AssertionError('unexpected mesh load'))
    monkeypatch.setattr(geometry,'load_stl_triangles',load)
    with pytest.raises(RuntimeError,match='bin_pose_required'):
        geometry.plan_scene_checked_place(node,[],np.eye(3),None,None,None,[.9,0,.75],lambda _: '/not-used')
    load.assert_not_called()


def test_real_obstacle_uses_registered_material_pose_and_combined_uncertainty_margin():
    scene=fitted_scene();frame=np.asarray(scene['rotation']);origin=np.asarray(scene['origin'])
    obstacle=geometry.NominalBinObstacle(scene['floor_center'],OUTER_BOUNDS,
        cavity_bounds=CAVITY_BOUNDS,bin_scene=scene)
    np.testing.assert_array_equal(obstacle.origin,origin);np.testing.assert_array_equal(obstacle.rotation,frame)
    assert obstacle.margin==pytest.approx(.005+scene['registration_margin_m'])
    local=(obstacle.corners-origin)@frame
    np.testing.assert_allclose(local.min(axis=0),np.asarray(OUTER_BOUNDS[0])-obstacle.margin)
    cube=geometry._box_triangles([.001,.001,.001]);transform=np.eye(4);transform[:3,:3]=frame
    transform[:3,3]=origin+frame@[OUTER_BOUNDS[1][0]+obstacle.margin*.5,0,0]
    assert obstacle.intersects(cube,transform,True)
    transform[:3,3]=origin+frame@[OUTER_BOUNDS[1][0]+obstacle.margin+.005,0,0]
    assert not obstacle.intersects(cube,transform,True)
    legacy=geometry.NominalBinObstacle(scene['floor_center'],OUTER_BOUNDS,cavity_bounds=CAVITY_BOUNDS)
    assert not np.allclose(legacy.rotation,obstacle.rotation)


class ValidationBoundary(RuntimeError): pass


def parameter_only_node(monkeypatch, cls, overrides, *, stop_parameter=None):
    monkeypatch.setattr(Node,'__init__',lambda *a,**kw:None)
    node=object.__new__(cls);values={}
    node.declare_parameter=lambda name,value:values.setdefault(name,overrides.get(name,value))
    def get(name):
        if name==stop_parameter:raise ValidationBoundary()
        return SimpleNamespace(value=values[name])
    node.get_parameter=get
    return node,values


def test_actual_manipulation_initialization_checks_dependencies_and_keeps_default_off(monkeypatch):
    for overrides,accepted in [({},True),({'bin_scene_required':True},False),
            ({'bin_scene_required':True,'table_scene_required':True},False),
            ({'bin_scene_required':True,'book_centered_place_enabled':True},False),
            ({'bin_scene_required':True,'table_scene_required':True,'book_centered_place_enabled':True},True)]:
        node,values=parameter_only_node(monkeypatch,manipulation.ManipulationNode,overrides,
                                       stop_parameter='scene_cartesian_wall_seconds')
        with pytest.raises(ValidationBoundary if accepted else ValueError,
                           match=None if accepted else 'bin_scene_required requires'):
            manipulation.ManipulationNode.__init__(node)
        if not overrides:assert values['bin_scene_required'] is False


def test_actual_perception_initialization_checks_dependency_and_keeps_default_off(monkeypatch,tmp_path):
    monkeypatch.setattr(perception,'load_digit_templates',lambda path:{})
    monkeypatch.setattr(perception,'Buffer',lambda:(_ for _ in ()).throw(ValidationBoundary()))
    for override,accepted in [({},True),({'bin_scene_enabled':True},False),
            ({'bin_scene_enabled':True,'table_scene_enabled':True},True)]:
        node,values=parameter_only_node(monkeypatch,perception.PerceptionNode,
                                       dict(override,image_output_dir=str(tmp_path)))
        with pytest.raises(ValidationBoundary if accepted else ValueError,
                           match=None if accepted else 'bin_scene_enabled requires'):
            perception.PerceptionNode.__init__(node)
        if not override:assert values['bin_scene_enabled'] is False
