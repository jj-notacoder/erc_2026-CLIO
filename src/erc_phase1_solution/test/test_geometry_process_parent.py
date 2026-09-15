"""Actual parked-scene predicate and process-prefix adapter behavior."""
import copy
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from erc_phase1_solution.geometry_process_parent import PairedSceneCapture, ProcessPlaceChecker
from erc_phase1_solution.geometry_process_pool import GeometryProcessError
from erc_phase1_solution.pure_geometry_owner import HEAD_JOINTS
from erc_phase1_solution.motion_profiles import RIGHT_ARM_JOINTS
from test_geometry_process_protocol import delta, state


def sensor_fixture(transition_samples=3):
    names=(*RIGHT_ARM_JOINTS,*HEAD_JOINTS)
    node=SimpleNamespace(_cancel=threading.Event(),_lock=threading.Lock(),
        joints={name:0. for name in names},_joint_stamps_ns={name:1_000_000_000 for name in names},
        _staging_odom=dict(stamp_ns=1_000_000_000,pose=[0.,0.,0.],linear_speed=0.,angular_speed=0.),
        right_chain=SimpleNamespace(active_names=('torso_lift_joint',*RIGHT_ARM_JOINTS)),
        head_chain=SimpleNamespace(active_names=('torso_lift_joint',*HEAD_JOINTS)),chain=object(),
        carried_collision_meshes=(),carried_transition_samples=transition_samples,_held_book_corners=np.zeros((8,3)),
        _selected_place_scene_reference=dict(base_pose=[0.,0.,0.],parked_joints={name:0. for name in names}),
        _selected_place_bin_scene=dict(id='bin'),_selected_place_table_scene=dict(id='table'))
    def clock():
        assert not node._lock.locked(), 'live lock held while reading clock'
        return SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=1_000_000_000))
    node.get_clock=clock
    current=state();current.node=node;current.attached=np.zeros((8,3))
    description=dict(bin_scene=copy.deepcopy(node._selected_place_bin_scene),
        table_scene=copy.deepcopy(node._selected_place_table_scene),attached_corners=np.zeros((8,3)).tolist())
    capture=PairedSceneCapture(node,current,description)
    pool=SimpleNamespace(epoch=1,identity=SimpleNamespace(source_id='1'*64,model_id='2'*64),scene_id='3'*64)
    return node,current,capture,pool


def test_capture_uses_one_owned_joint_and_odom_epoch_with_no_ipc_under_lock():
    node,_,capture,pool=sensor_fixture()
    def clock():
        assert not node._lock.locked()
        node.joints={name:.0009 for name in node.joints}
        node._staging_odom['pose'][0]=.001
        return SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=1_000_000_000))
    node.get_clock=clock
    q=np.arange(8,dtype=float)
    first=capture.capture(0,(q,.018,True),pool=pool)
    q[:]=99.
    assert np.array_equal(np.frombuffer(first.q,dtype=np.float64),np.arange(8))
    assert np.all(np.frombuffer(first.right,dtype=np.float64)==0.)
    assert np.all(np.frombuffer(first.head,dtype=np.float64)==0.)
    second=capture.capture(1,(np.zeros(8),.069,False),pool=pool)
    assert np.all(np.frombuffer(second.right,dtype=np.float64)==.0009)
    assert np.all(np.frombuffer(second.head,dtype=np.float64)==.0009)
    assert second.aperture==.069 and second.loaded is False


@pytest.mark.parametrize('kind',['joint_stale','joint_future','joint_nonfinite','base_stale',
    'base_moving','base_drift','head_drift','right_drift'])
def test_original_freshness_motion_and_reference_limits_veto_enqueue(kind):
    node,_,capture,pool=sensor_fixture()
    if kind=='joint_stale':node._joint_stamps_ns[HEAD_JOINTS[0]]=600_000_000
    elif kind=='joint_future':node._joint_stamps_ns[HEAD_JOINTS[0]]=1_051_000_000
    elif kind=='joint_nonfinite':node.joints[HEAD_JOINTS[0]]=float('nan')
    elif kind=='base_stale':node._staging_odom['stamp_ns']=600_000_000
    elif kind=='base_moving':node._staging_odom['linear_speed']=.006
    elif kind=='base_drift':node._staging_odom['pose'][0]=.0021
    elif kind=='head_drift':node.joints[HEAD_JOINTS[0]]=.0011
    else:node.joints[RIGHT_ARM_JOINTS[0]]=.0011
    with pytest.raises(RuntimeError):capture.capture(0,(np.zeros(8),.018,True),pool=pool)
    assert capture.last_admission is None


@pytest.mark.parametrize('kind',['reference_replaced','reference_edited','bin_replaced','table_edited',
    'model_replaced','chain_replaced','held_replaced','held_edited','sampling_changed'])
def test_selected_scene_and_geometry_generation_cannot_change_under_pool(kind):
    node,_,capture,pool=sensor_fixture()
    if kind=='reference_replaced':node._selected_place_scene_reference=copy.deepcopy(node._selected_place_scene_reference)
    elif kind=='reference_edited':node._selected_place_scene_reference['base_pose'][0]=.001
    elif kind=='bin_replaced':node._selected_place_bin_scene=copy.deepcopy(node._selected_place_bin_scene)
    elif kind=='table_edited':node._selected_place_table_scene['id']='other'
    elif kind=='model_replaced':node.carried_collision_meshes=(object(),)
    elif kind=='chain_replaced':node.chain=object()
    elif kind=='held_replaced':node._held_book_corners=np.zeros((8,3))
    elif kind=='sampling_changed':node.carried_transition_samples=11
    else:node._held_book_corners[0,0]=.00001
    with pytest.raises(GeometryProcessError,match='generation changed'):
        capture.capture(0,(np.zeros(8),.018,True),pool=pool)


def test_capture_cancellation_prevents_even_an_identical_cached_query():
    node,_,capture,pool=sensor_fixture();node._cancel.set()
    with pytest.raises(GeometryProcessError,match='cancelled'):
        capture.capture(0,(np.zeros(8),.018,True),pool=pool)


class RecordingPool:
    def __init__(self,binding,reject_index=None):
        self.__dict__.update(binding.__dict__)
        self.queries=[];self.reject_index=reject_index
    def evaluate_sequence(self,items,*,capture,consume,is_cached):
        for item in items:
            index=len(self.queries);q=capture(index,item);self.queries.append(q)
            current=None if is_cached(q) else delta(q,accepted=index!=self.reject_index,minimum=1.-index*.001)
            if not consume(q,current):return False
        return True


@pytest.mark.parametrize('distance,steps,expected',[(0.,3,1),(.02,3,3),(.121,3,8),(.01,11,11)])
def test_leg_keeps_endpoint_count_spacing_and_current_aperture(distance,steps,expected):
    node,current,capture,binding=sensor_fixture(steps)
    pool=RecordingPool(binding);checker=ProcessPlaceChecker(current,pool,capture)
    first=np.zeros(8);last=first.copy();last[2]=distance
    assert checker.leg(first,last,.018,True)
    actual=np.array([np.frombuffer(q.q,dtype=np.float64) for q in pool.queries])
    assert len(actual)==expected
    assert np.array_equal(actual,np.array([first+(last-first)*f for f in np.linspace(0.,1.,expected)]))
    assert all(q.aperture==.018 and q.loaded is True for q in pool.queries)


def test_leg_first_rejection_fraction_and_prefix_minimum_are_committed():
    _,current,capture,binding=sensor_fixture();pool=RecordingPool(binding,reject_index=2)
    checker=ProcessPlaceChecker(current,pool,capture)
    first=np.zeros(8);last=first.copy();last[2]=.121
    assert not checker.leg(first,last,.069,False)
    assert current.last_rejection==dict(reason='robot_self',fraction=2/7)
    assert current.samples==3 and current.minimum_moving_left_z==.998
    assert all(q.aperture==.069 and q.loaded is False for q in pool.queries)
