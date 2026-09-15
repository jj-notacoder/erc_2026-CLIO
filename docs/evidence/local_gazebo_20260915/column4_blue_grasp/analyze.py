#!/usr/bin/env python3
"""Reproduce this trial's offline pose comparison; never connects to ROS/Gazebo.

Requires the repository's NumPy/SciPy kinematics dependencies. The default input
is the small retained pose subset. --full-pose-log verifies and analyzes the
original local log instead. Separate pose/joint stamps remain separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
META = json.loads((HERE / 'provenance.json').read_text())
NAMES = ['torso_lift_joint', *[f'arm_left_{i}_joint' for i in range(1, 8)]]
BOOK = 'book_col_5_row_4_blue'


def rotation(q):
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def transform(pose):
    value = np.eye(4)
    value[:3, :3] = rotation(pose['orientation_xyzw'])
    value[:3, 3] = pose['position']
    return value


def angle(a, b):
    return float(np.arccos(np.clip((np.trace(a.T @ b)-1)/2, -1, 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-pose-log', type=Path)
    args = parser.parse_args()
    # Fail if the FK/model description differs from this trial's recorded source.
    for name in ('src/erc_description/urdf/tiago_pro.urdf',
                 'src/erc_description/models/book/sdf/erc_book.sdf',
                 'src/erc_phase1_solution/erc_phase1_solution/kinematics.py'):
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != META['source_hashes'][name]:
            raise SystemExit(f'Recorded source hash mismatch: {name}')
    sys.path.insert(0, str(ROOT/'src/erc_phase1_solution'))
    from erc_phase1_solution.kinematics import URDFChain
    chain = URDFChain.from_urdf(ROOT/'src/erc_description/urdf/tiago_pro.urdf',
                               'base_footprint', 'gripper_left_grasping_link', NAMES)
    source = args.full_pose_log or HERE/'selected_pose_records.jsonl'
    if args.full_pose_log and hashlib.sha256(source.read_bytes()).hexdigest() != META['original_log_hashes']['pick_pose_diagnostics.jsonl']:
        raise SystemExit('Original pose log hash mismatch')
    records = [json.loads(line) for line in source.read_text().splitlines()]
    records = [r for r in records if r.get('world') and r.get('joint_feedback')]
    journal = [json.loads(line) for line in (HERE/'selected_journal_events.jsonl').read_text().splitlines()]
    plan = next(e['payload'] for e in journal if e.get('payload', {}).get('event') == 'pick_approach_planned')
    target = np.array(plan['target'])
    commanded = chain.forward(plan['solutions'][-1])
    assert np.allclose(commanded[:3, 3], plan['actual_grasp_position'], atol=1e-12)
    assert np.allclose(commanded[:3, :3], plan['actual_grasp_orientation'], atol=1e-12)

    def derive(record):
        world = record['world']; joint = record['joint_feedback']
        robot = transform(world['model_poses']['tiago_pro'])
        book = transform(world['model_poses'][BOOK])
        # URDF-to-SDF audit: model frame and base_footprint coincide. Base_link
        # is +0.0762 m above base_footprint and is already included in chain FK.
        relative = np.linalg.inv(robot) @ book
        # Book's local box is .25 x .02 x .16; model/link/collision coincide.
        # Front is local -Z (.08 m), not -X: the spawned book is pitch pi/2.
        front = (relative @ np.array([0., 0., -.08, 1.]))[:3]
        inward = relative[:3, 2]
        qmap = dict(zip(joint['names'], joint['positions']))
        q = np.array([qmap[name] for name in NAMES])
        hand = chain.forward(q)
        hand_world = robot @ hand  # Cross-stamp diagnostic, not a simultaneous sample.
        book_in_hand = np.linalg.inv(hand_world) @ book
        signs = np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        corners = book[:3, 3] + (signs*np.array([.125,.01,.08])) @ book[:3, :3].T
        velocity = dict(zip(joint['names'], joint['velocities']))
        return dict(
            receipt_ros_ns=record['ros_clock_ns'],
            pose_stamp_ns=world['pose_stamp_ns'], joint_stamp_ns=joint['stamp_ns'],
            pose_minus_joint_stamp_ns=world['pose_stamp_ns']-joint['stamp_ns'],
            status_event=(record.get('status') or {}).get('event'),
            book_center_world_m=book[:3,3].tolist(),
            book_world_min_max_z_m=[float(corners[:,2].min()),float(corners[:,2].max())],
            book_center_base_m=relative[:3,3].tolist(), book_front_base_m=front.tolist(),
            target_minus_book_front_base_m=(target-front).tolist(),
            target_distance_to_book_front_plane_m=float((target-front) @ inward),
            measured_hand_base_m=hand[:3,3].tolist(),
            measured_hand_minus_commanded_hand_m=(hand[:3,3]-commanded[:3,3]).tolist(),
            measured_hand_orientation_error_rad=angle(hand[:3,:3],commanded[:3,:3]),
            measured_hand_depth_from_book_front_plane_m=float((hand[:3,3]-front) @ inward),
            book_center_in_grasp_frame_m=book_in_hand[:3,3].tolist(),
            maximum_arm_joint_error_rad=float(np.max(np.abs(q[1:]-np.array(plan['solutions'][-1][1:])))),
            maximum_arm_speed_rad_s=max(abs(velocity.get(name,0.)) for name in NAMES[1:]),
            book_inward_axis_vs_base_x_rad=angle(relative[:3,:3],rotation([0,np.sqrt(.5),0,np.sqrt(.5)])),
            gripper_master_position_m=qmap['gripper_left_finger_joint'])

    moments = {}
    for label, event, stamp in [('reacquisition',None,35.872), ('before_close',None,119.75),
                               ('gripper_closed','gripper_closed',0),
                               ('geometry_verified','lift_first_measured_geometry_verified',0),
                               ('retention_verified','retention_verified',0),
                               ('terminal_observation',None,121.5)]:
        candidates = [r for r in records if (r.get('status')or{}).get('event') == event] if event else records
        record = candidates[0] if event else min(candidates,key=lambda r:abs(r['joint_feedback']['stamp_ns']-stamp*1e9))
        moments[label] = derive(record)
    # Deduplicate cached repeats of the exact same Gazebo pose timestamp.
    stationary = {r['world']['pose_stamp_ns']:r for r in records if 34e9 <= r['world']['pose_stamp_ns'] <= 121.5e9}
    ordered = [stationary[t] for t in sorted(stationary)]
    values = [derive(r) for r in ordered]
    centers = np.array([v['book_center_world_m'] for v in values])
    rotations = [rotation(r['world']['model_poses'][BOOK]['orientation_xyzw']) for r in ordered]
    minmaxz = np.array([v['book_world_min_max_z_m'] for v in values])
    output = dict(trial_id=META['trial_id'],
        input_scope='verified full original pose log' if args.full_pose_log else 'retained selected pose records',
        frame_assumption='Model=base_footprint confirmed by offline URDF-to-SDF conversion; see frame_audit.json. Model/book link/collision coincide.',
        pairing_scope='Model-to-model geometry shares a timestamp. Joint FK is separately stamped; no interpolation or simultaneity claim.',
        planned_front_base_m=plan['target'], planned_grasp_base_m=plan['grasp'],
        planned_joint_fk_base_m=commanded[:3,3].tolist(),
        moments=moments,
        observed_book_motion=dict(unique_world_samples=len(ordered),
            first_pose_stamp_ns=ordered[0]['world']['pose_stamp_ns'],
            last_pose_stamp_ns=ordered[-1]['world']['pose_stamp_ns'],
            final_minus_first_center_world_m=(centers[-1]-centers[0]).tolist(),
            maximum_center_displacement_from_first_m=float(np.max(np.linalg.norm(centers-centers[0],axis=1))),
            center_axis_peak_to_peak_m=np.ptp(centers,axis=0).tolist(),
            maximum_attitude_change_from_first_rad=max(angle(rotations[0],r) for r in rotations),
            bottom_corner_z_range_m=[float(minmaxz[:,0].min()),float(minmaxz[:,0].max())],
            top_corner_z_range_m=[float(minmaxz[:,1].min()),float(minmaxz[:,1].max())]))
    print(json.dumps(output,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
