"""Official-geometry PLACE replay with explicit recorded/nominal input provenance."""
import argparse
from collections import deque
from copy import deepcopy
import json
import yaml
from pathlib import Path
import signal
import sys
import hashlib
import threading
import time
from types import SimpleNamespace as NS

WORKSPACE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKSPACE/'src/erc_phase1_solution'))
sys.path.insert(0, str(WORKSPACE/'src/erc_phase1_solution/test'))

import numpy as np
from ament_index_python.packages import get_package_share_directory
from test_shutdown import _official_manipulation_planner
from erc_phase1_solution import scene_checked_place as planner
from erc_phase1_solution.adaptive_grasp import GripperFeedback
from erc_phase1_solution.book_centered_place import book_centered_place_target
from erc_phase1_solution.lower_shelf_carry import POSITIVE_WRIST_CARRY
from erc_phase1_solution.motion_profiles import IK_JOINTS, RIGHT_ARM_JOINTS, RIGHT_HOME
from erc_phase1_solution.place_contact_guard import PlaceContactGuard
from erc_phase1_solution.placement_scene_context import measured_scene_context
from erc_phase1_solution.release_only_place_planning import capture as capture_release_request
from erc_phase1_solution.shelf_cradle_geometry import ShelfCradleGeometry
from erc_phase1_solution.exact_local_bounds import ModelLocalMesh
from erc_phase1_solution.kinematics import CollisionMesh

ROOT=WORKSPACE
EVIDENCE=ROOT/'docs/evidence/local_gazebo_20260915/column4_blue_transport'


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,(list,tuple)):return [plain(x) for x in value]
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--scene-stamp',type=int,default=115500000000)
    parser.add_argument('--trace',action='store_true')
    args=parser.parse_args()
    label=f'positive_registered_place_{args.scene_stamp}'+('_trace' if args.trace else '')
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(__file__).with_name('source_manifest.json').read_text())
    hashes={name:hashlib.sha256((ROOT/'src/erc_phase1_solution/erc_phase1_solution'/name).read_bytes()).hexdigest() for name in manifest}
    if any(hashes[name]!=row['draft_sha256'] for name,row in manifest.items()):
        raise RuntimeError('runtime source differs from pinned evidence; review changes before replay')
    profile=yaml.safe_load((ROOT/'src/erc_phase1_solution/config/collision_quality.yaml').read_text())['erc_manipulation']['ros__parameters']
    place_height=float(profile['place_torso_height'])
    result=dict(mode='integrated_positive',runtime_source_sha256=hashes,scene_observation_stamp_ns=args.scene_stamp,
        exact_live_replay=False,controller_calls=[],assumptions=[
            '115.500 scene selection inferred from independent recorder ordering, not proven selected',
            'Recorded scene numerical coordinates retained; capture frame movement not logged',
            'Carry start uses final commanded POSITIVE_WRIST_CARRY; actual measured endpoints unlogged',
            'Right/head use commanded parked RIGHT_HOME and[0,-.60], not unlogged measured feedback',
            'Held corners reconstructed by production attachment from recorded target/grasp IK',
            'Staging seed recomputed by same production Cartesian staging solves from recorded lift terminal',
            'Replay clock and fresh feedback stamps are fixture values, not a live freshness proof'])
    started=time.monotonic();events=[]
    def emit(event,**fields):
        row=dict(event=event,elapsed_seconds=time.monotonic()-started,**fields)
        events.append(row);print(json.dumps(plain(row)),flush=True)
    node=None
    try:
        records=[json.loads(line) for line in (EVIDENCE/'recorded_bin_scenes.jsonl').read_text().splitlines()]
        raw=next(r for r in records if r['payload']['observation_stamp_ns']==args.scene_stamp)
        recorded=raw['payload'];result['scene_record']=raw
        journal=[json.loads(line) for line in (EVIDENCE/'selected_journal_events.jsonl').read_text().splitlines()]
        approach=next(r['payload'] for r in journal if r.get('payload',{}).get('event')=='pick_approach_planned')
        lift=next(r['payload'] for r in journal if r.get('payload',{}).get('event')=='lift_first_extraction_planned')
        contact=next(r['payload'] for r in journal if r.get('payload',{}).get('event')=='retention_verified' and r['payload'].get('phase')=='post_navigation')
        node=_official_manipulation_planner();node._lock=threading.RLock();node._cancel=threading.Event()
        def timeout(*args):
            node._cancel.set()
            raise TimeoutError('external offline replay180second cap')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(180)
        node.carried_book_dimensions=np.array([.16,.02,.25]);node.gripper_open=.069
        node._held_book_corners=node._attached_book_corners(approach['target'],approach['solutions'][-1])
        node._gravity_supported_payload=True
        carried=POSITIVE_WRIST_CARRY.copy();torso_ready=carried.copy();torso_ready[0]=place_height
        node.place_torso_height=place_height;node.scene_cartesian_wall_seconds=1800.
        node.joints.update(zip(IK_JOINTS,carried));node.joints.update(zip(RIGHT_ARM_JOINTS,RIGHT_HOME))
        node.joints.update(head_1_joint=0.,head_2_joint=-.60,gripper_left_finger_joint=contact['measured_position'])
        node._joint_stamps_ns={name:1_000_000_000 for name in node.joints}
        node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=1_000_000_000))
        node._staging_odom=dict(stamp_ns=1_000_000_000,pose=recorded['scene_base_reference']['pose'],
            linear_speed=0.,angular_speed=0.,frame_id='odom',child_frame_id='base_footprint')
        node._gripper_feedback_samples=deque([GripperFeedback(1_000_000_000,contact['measured_position'],0.)])
        urdf=Path(get_package_share_directory('erc_description'))/'urdf/tiago_pro.urdf'
        node._shelf_cradle_geometry=ShelfCradleGeometry(urdf,get_package_share_directory,immutable_local=True)
        meshes=[]
        for mesh in node.carried_collision_meshes:
            local=ModelLocalMesh(mesh.triangles)
            meshes.append(CollisionMesh(mesh.link,local.snapshot()[0],mesh.bounds,mesh.watertight,local))
        node.carried_collision_meshes=tuple(meshes)
        extension=node._solve_post_retreat_clearance_extension(lift['terminal'],extension_distance=.12)
        rolled=node._solve_carried_cradle(extension[-1])
        _,raised,retracted=node._solve_supported_post_retreat_staging(rolled,vertical_offset=.18)
        staging=retracted[-1].copy();staging[0]=place_height
        node._carried_staging_solution=staging
        node._publish_status=lambda event,**fields:emit(event,**fields)
        for name in ('_move_arm','_move_torso','_move_head','_move_gripper','_follow'):
            def forbidden(*args,**kwargs):
                result['controller_calls'].append(str(args));raise AssertionError('offline controller dispatch')
            setattr(node,name,forbidden)
        identity=dict(trial_id='dd540101cd5e',placement_attempt_id='fa0a8f56ac8243e49ff520852adae987',target_model='book_col_5_row_4_blue')
        payload=dict(identity,completion_mode='release_pose')
        node._busy=True;node._target_book_model=identity['target_model'];node._goal_handles=[]
        node._pending_retained_acceptances=[];node._release_pose_owner=None
        for name in ('release_only_place_planning_enabled','place_finish_at_release_enabled','delivery_evidence_enabled','book_centered_place_enabled','table_scene_required','bin_scene_required'):
            setattr(node,name,True)
        node.bin_clearance_timing_enabled=False
        node._selected_place_scene_reference=measured_scene_context(node)
        node._active_place_scene_reference=node._selected_place_scene_reference
        node._selected_place_bin_scene=deepcopy(recorded['bin_scene'])
        node._selected_place_table_scene=deepcopy(recorded['table_scene'])
        node._place_contact_guard=PlaceContactGuard(1_000_000_000,identity)
        request=capture_release_request(node,payload)
        floor=np.asarray(recorded['bin_scene']['floor_center'])
        target=book_centered_place_target(node._held_book_corners,floor,.27,bin_rotation=recorded['bin_scene']['rotation'])
        release=target.tool_position;above=release.copy();above[2]+=.23
        clearance=above.copy();clearance[0]=.60
        positions=[clearance,*node._interpolate_positions(clearance,above,.06),*node._interpolate_positions(above,release,.06)]
        result['inputs']=dict(front=approach['target'],grasp=approach['solutions'][-1],attached=node._held_book_corners,
            selected_place_height=place_height,carry=carried,torso_ready=torso_ready,staging=staging,positions=positions,rotation=target.tool_rotation,
            bin_floor=floor,aperture=contact['measured_position'])
        emit('INPUTS',**result['inputs'])
        original=planner.solve_scene_cartesian
        def solve(*args2,**kwargs):
            try:return original(*args2,**kwargs)
            except Exception as error:
                result['search_failure']=getattr(error,'diagnostics',None)
                raise
        planner.solve_scene_cartesian=solve
        if args.trace:
            original_ik=node.chain.solve
            result['ik_trace']=[]
            def traced_ik(matrix,seeds,**options):
                q,score=original_ik(matrix,seeds,**options)
                result['ik_trace'].append(dict(position=matrix[:3,3].tolist(),
                    seed=np.asarray(seeds[0]).tolist(), options=options,
                    q=None if q is None else np.asarray(q).tolist(),score=float(score)))
                return q,score
            node.chain.solve=traced_ik
        phase=time.monotonic()
        plan=planner.plan_scene_checked_place(node,positions,target.tool_rotation,carried,torso_ready,
            staging,floor,get_package_share_directory,table_scene=node._selected_place_table_scene,
            bin_scene=node._selected_place_bin_scene,release_only_request=request)
        result.update(passed=True,planning_seconds=time.monotonic()-phase,
            plan=dict(solutions=plan.solutions,setup=plan.setup,diagnostics=plan.diagnostics,
                release_endpoint=plan.release_only_endpoint.goal))
        emit('PASSED',planning_seconds=result['planning_seconds'],diagnostics=plan.diagnostics)
    except BaseException as error:
        result.update(passed=False,error_type=type(error).__name__,error=str(error))
        emit('FAILED',error_type=type(error).__name__,error=str(error),diagnostics=result.get('search_failure'))
    finally:
        signal.alarm(0)
        result.update(total_seconds=time.monotonic()-started,events=events)
        (output/(label+'.json')).write_text(json.dumps(plain(result),indent=2))
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
