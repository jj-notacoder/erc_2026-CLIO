from types import SimpleNamespace
import threading
import pytest

from erc_phase1_solution.placement_scene_context import measured_scene_context, matching_table_scene


def node_fixture():
    names = ['arm_right_1_joint', 'head_1_joint']
    node = SimpleNamespace(_lock=threading.Lock(), joints=dict.fromkeys(names, 0.),
        _joint_stamps_ns=dict.fromkeys(names, 1_000_000_000),
        _staging_odom=dict(stamp_ns=1_000_000_000, pose=[1., 2., 3.], linear_speed=0., angular_speed=0.),
        right_chain=SimpleNamespace(active_names=('torso_lift_joint', names[0])),
        head_chain=SimpleNamespace(active_names=('torso_lift_joint', names[1])),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_010_000_000)))
    return node


def test_registered_context_allows_active_torso_arm_hand_but_rejects_parked_motion():
    node = node_fixture(); reference = measured_scene_context(node)
    node.joints.update(torso_lift_joint=.3, arm_left_1_joint=.5, gripper_left_finger_joint=.069)
    assert measured_scene_context(node, reference)['parked_joints'] == reference['parked_joints']
    node.joints['head_1_joint'] = .002
    with pytest.raises(RuntimeError, match='parked_geometry_moved'):
        measured_scene_context(node, reference)


@pytest.mark.parametrize('change,reason', [
    (lambda n: n._staging_odom.update(stamp_ns=0), 'stale_odometry'),
    (lambda n: n._joint_stamps_ns.update(head_1_joint=0), 'stale_joint'),
    (lambda n: n._staging_odom.update(linear_speed=.006), 'not_stationary'),
    (lambda n: n._staging_odom.update(pose=[1.003, 2., 3.]), 'base_moved'),
    (lambda n: n._staging_odom.update(pose=[1., 2., 3.006]), 'base_moved'),
])
def test_scene_rejects_stale_speed_and_pose_drift(change, reason):
    node = node_fixture(); reference = measured_scene_context(node); change(node)
    with pytest.raises(RuntimeError, match=reason):
        measured_scene_context(node, reference)


def test_table_requires_exact_producer_and_same_transformed_bin_point():
    entry = dict(observation_stamp_ns=123, table_scene=dict(valid=True, frame='base_footprint'), bin_floor_point_base=[.8, 0., .75])
    assert matching_table_scene(entry, 123, [.8, 0., .75]) is entry['table_scene']
    with pytest.raises(RuntimeError, match='matching_table'):
        matching_table_scene(entry, 124, [.8, 0., .75])
    with pytest.raises(RuntimeError, match='frame_mismatch'):
        matching_table_scene(entry, 123, [.81, 0., .75])
    entry['table_scene'] = dict(valid=False, reason='no_two_edges')
    with pytest.raises(RuntimeError, match='no_two_edges'):
        matching_table_scene(entry, 123, [.8, 0., .75])
