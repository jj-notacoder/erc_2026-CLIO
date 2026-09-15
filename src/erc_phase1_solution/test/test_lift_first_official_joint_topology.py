"""Actual official gripper topology, distinct from mocked motion-order fixtures."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('rclpy')
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import JointState

from erc_phase1_solution.shelf_cradle_geometry import ShelfCradleGeometry
from erc_phase1_solution.rigid_palm_preflight import FINGER_COLLISION_LINKS, PALM_COLLISION_LINK
from test_lift_first_integration import _official_actuated_node, _OFFICIAL_ACTUATED_NAMES


@pytest.fixture(scope='module')
def official_geometry():
    urdf=Path(get_package_share_directory('erc_description'))/'urdf'/'tiago_pro.urdf'
    return ShelfCradleGeometry(urdf,get_package_share_directory)


def _node_from_official_joint_message(geometry, omitted=()):
    node=_official_actuated_node()
    node._shelf_cradle_geometry=geometry
    node.joints={};node._joint_stamps_ns={}
    message=JointState()
    message.header.stamp.sec=1
    message.name=[name for name in _OFFICIAL_ACTUATED_NAMES if name not in omitted]
    message.position=[.0183 if name=='gripper_left_finger_joint' else 0. for name in message.name]
    message.velocity=[0.]*len(message.name)
    message.effort=[0.]*len(message.name)
    node._on_joint_state(message)
    return node


def test_real_collision_chains_omit_master_but_its_measured_value_drives_mimics(official_geometry):
    geometry=official_geometry
    assert set(geometry.chains)=={PALM_COLLISION_LINK,*FINGER_COLLISION_LINKS}
    ancestors={name for chain in geometry.chains.values() for name in chain.active_names}
    # This physical URDF fact caused Run14's false rejection: the screw branch
    # has no collision mesh, while mesh-hinge <mimic> dependencies name master.
    assert 'gripper_left_finger_joint' not in ancestors
    passive={name for name in ancestors if name.startswith('gripper_left_')}
    assert len(passive)==6
    assert all(geometry.mimics[name].get('joint')=='gripper_left_finger_joint' for name in passive)
    node=_node_from_official_joint_message(geometry)
    assert len(node.joints)==23
    sample=node._lift_first_measurements()
    assert sample['fingers']=={'gripper_left_finger_joint':.0183}
    assert set(sample['modeled_finger_joints'])==passive
    assert sample['modeled_tool_allowance_m']==.005
    assert sample['finger_geometry_basis']=='measured_master_with_official_urdf_mimics'
    actual_master_model=geometry.local_surfaces(.0183,sample['fingers'])
    zero_master_model=geometry.local_surfaces(0.)
    assert set(actual_master_model)==set(geometry.chains)
    assert all(np.all(np.isfinite(surface)) for surface in actual_master_model.values())
    assert not np.allclose(actual_master_model['gripper_left_fingertip_left_link'],
                           zero_master_model['gripper_left_fingertip_left_link'])


def test_real_collision_topology_still_requires_measured_master(official_geometry):
    node=_node_from_official_joint_message(official_geometry,omitted=('gripper_left_finger_joint',))
    with pytest.raises(RuntimeError,match='measurement stale: gripper_left_finger_joint'):
        node._lift_first_measurements()


def test_real_collision_topology_rejects_stale_master(official_geometry):
    node=_node_from_official_joint_message(official_geometry)
    node._joint_stamps_ns['gripper_left_finger_joint']=0
    with pytest.raises(RuntimeError,match='measurement stale: gripper_left_finger_joint'):
        node._lift_first_measurements()
