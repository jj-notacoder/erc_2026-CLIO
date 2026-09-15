"""One complete official-model bottom plan; no commands or simulator process."""
import json
from pathlib import Path
import threading
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np

from ament_index_python.packages import get_package_share_directory
from test_shutdown import _official_manipulation_planner
from erc_phase1_solution import lower_shelf_pick
from erc_phase1_solution import shelf_cradle_geometry as cradle
from erc_phase1_solution.empty_pickup_collision import EmptyPickupCollision
from erc_phase1_solution.empty_shelf_bounds import EmptyShelfBounds
from erc_phase1_solution.exact_local_bounds import ModelLocalMesh
from erc_phase1_solution.kinematics import CollisionMesh
from erc_phase1_solution.lift_first_extraction import RelativeShelfBay, plan_lift_first_extraction
from erc_phase1_solution.motion_profiles import HOME, IK_JOINTS, RIGHT_ARM_JOINTS


def test_official_bottom_complete_registered_entry_lift_carry_recovery(monkeypatch):
    fixture=json.loads((Path(__file__).parent/'fixtures'/'bottom_registered_nominal.json').read_text())
    node=_official_manipulation_planner()
    node.joints.update(zip(IK_JOINTS,fixture['initial_left']))
    node.joints.update(zip(RIGHT_ARM_JOINTS,fixture['right']))
    node.joints.update(zip(('head_1_joint','head_2_joint'),fixture['head']))
    node.joints['gripper_left_finger_joint']=.069
    node.carried_book_dimensions=np.array(fixture['book_dimensions'])
    node.gripper_open=.069;node.pregrasp_offset=.14;node.pick_torso_height=.35
    node.pick_position_tolerance=.0005;node.pick_orientation_tolerance=.01
    node._cancel=threading.Event();node._lock=threading.Lock()
    node._joint_stamps_ns={name:1_000_000_000 for name in node.joints}
    node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=1_000_000_000))
    node._publish_status=Mock()
    node._move_arm_solution=Mock(side_effect=AssertionError('planner dispatched arm motion'))
    node._move_torso=Mock(side_effect=AssertionError('planner dispatched torso motion'))
    node._command_gripper=Mock(side_effect=AssertionError('planner dispatched gripper motion'))
    # The existing fixture loader supplies raw facets. Production enables these
    # immutable owners; wrapping preserves the exact official model bytes.
    meshes=[]
    for mesh in node.carried_collision_meshes:
        local=ModelLocalMesh(mesh.triangles)
        meshes.append(CollisionMesh(mesh.link,local.snapshot()[0],mesh.bounds,mesh.watertight,local))
    node.carried_collision_meshes=tuple(meshes)
    urdf=Path(get_package_share_directory('erc_description'))/'urdf'/'tiago_pro.urdf'
    node._shelf_cradle_geometry=cradle.ShelfCradleGeometry(urdf,get_package_share_directory,immutable_local=True)
    guard=EmptyPickupCollision.capture(node)
    front=np.array(fixture['front'])
    bay=RelativeShelfBay(marker_center_base=fixture['marker_center_base'],
        inward_axis_base=[1.,0.,0.],physical_column=3,lateral_uncertainty_m=.200,
        roof_uncertainty_m=.010,assume_upright_supported=True,
        source='independent fixed-marker official-model test fixture')
    tool_checks=[]
    original_tool=cradle.check_cradle_tool_route
    def checked_tool(*args,**kwargs):
        result=original_tool(*args,**kwargs)
        tool_checks.append((len(args[4]),result))
        return result
    monkeypatch.setattr(cradle,'check_cradle_tool_route',checked_tool)
    def lift(grasp,withdrawal):
        np.testing.assert_array_equal(grasp,fixture['grasp'])
        np.testing.assert_array_equal(withdrawal,fixture['withdrawal'])
        return plan_lift_first_extraction(node,front,grasp,withdrawal,bay=bay,
            aperture=.020,lift_m=.020,modeled_tool_allowance_m=.005)
    plan=lower_shelf_pick.plan_lower_shelf_pick(node,front,empty_guard=guard,lift_planner=lift,bay=bay)
    assert plan.candidate_name=='bottom_redundant_supported'
    assert plan.loaded_clearance_index is None
    np.testing.assert_array_equal(plan.solutions[-1],fixture['grasp'])
    np.testing.assert_array_equal(plan.extraction_solutions,fixture['withdrawal'])
    np.testing.assert_array_equal(plan.lift_plan.terminal,fixture['lift_terminal'])
    assert not plan.solutions[-1].flags.writeable
    assert not plan.extraction_solutions[0].flags.writeable
    assert node.pick_torso_height==.35
    cache=plan.cached_post_retreat_plan
    assert node._cached_post_retreat_plan is cache
    assert cache['compact_path']=='bottom' and cache['compact_radius']<=.45
    assert tool_checks==[(len(cache['legs']),None)]
    assert plan.unloaded_recovery_route
    np.testing.assert_array_equal(plan.unloaded_recovery_route[-1],HOME)
    assert guard.checked_samples>=61
    # Recheck the emitted forward route with the current shared shelf model,
    # independently of the backward proposal construction's traversal order.
    bounds=EmptyShelfBounds(node,front,plan.solutions[-1],guard,bay)
    assert bounds.normal_uncertainty_m==fixture['normal_uncertainty_m']
    assert bounds.plane_point[0]<=bounds.registered_lip[0]
    previous=guard.start.copy();elevated=previous.copy();elevated[0]=plan.pick_torso_height
    for goal in (elevated,*plan.transition_waypoints,plan.solutions[0]):
        assert bounds.edge(previous,goal,allow_entry=False) is None
        previous=goal
    for goal in plan.solutions[1:]:
        assert bounds.edge(previous,goal,allow_entry=True) is None
        previous=goal
    assert min(bounds.minimum_floor,bounds.minimum_roof,bounds.minimum_side,bounds.minimum_back)>0.
    previous=plan.lift_plan.terminal
    for goal,phase in cache['legs']:
        if phase not in ('post_retreat_clearance_extension','post_retreat_cradle_roll'):
            count=max(61,int(np.ceil(np.max(np.abs(goal-previous))/.02))+1)
            for q in np.linspace(previous,goal,count):
                assert node.chain.forward(q)[2,1]>=node.carried_supported_jaw_vertical_component
                assert -2.1-1e-9<=q[-1]<=1e-9
        previous=goal
    node._held_book_corners=cache['attached_corners']
    node._current_seed=lambda:cache['terminal'].copy()
    assert node._carried_head_transition_is_safe(0.,-.60)
    node._move_arm_solution.assert_not_called()
    node._move_torso.assert_not_called()
    node._command_gripper.assert_not_called()
