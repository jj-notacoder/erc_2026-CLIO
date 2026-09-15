"""Prepared offline geometry coverage; execution requires an explicit --run target.

These are SYNTHETIC onboard-aligned inputs, not recorded observations. Importing
this file performs no ROS, IK, mesh work, or controller actions. Runtime modules
are imported from the supplied production package; no draft method replacements.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args, scenario):
    # Deliberately deferred: --describe can run without importing ROS/numpy.
    package = Path(args.package).resolve()
    sys.path[:0] = [str(package), str(package / 'test')]
    import numpy as np
    from ament_index_python.packages import get_package_share_directory
    from test_shutdown import _official_manipulation_planner
    from erc_phase1_solution.motion_profiles import HOME, IK_JOINTS
    from erc_phase1_solution.empty_pickup_collision import EmptyPickupCollision
    from erc_phase1_solution.empty_shelf_bounds import EmptyShelfBounds
    from erc_phase1_solution.lift_first_extraction import (
        RelativeShelfBay, plan_lift_first_extraction, _bay_values,
    )
    from erc_phase1_solution.pickup_geometry_backend import (
        initialize_pickup_geometry_backend, run_pickup_geometry,
    )
    from erc_phase1_solution.shelf_cradle_geometry import ShelfCradleGeometry
    from erc_phase1_solution.exact_local_bounds import ModelLocalMesh
    from erc_phase1_solution.kinematics import CollisionMesh
    import erc_phase1_solution.lower_shelf_pick as helper

    expected_helper = package / 'erc_phase1_solution' / 'lower_shelf_pick.py'
    if Path(helper.__file__).resolve() != expected_helper.resolve():
        raise RuntimeError('coverage_requires_current_production_source_import')

    def plain(value):
        if isinstance(value, np.ndarray):
            return plain(value.tolist())
        if isinstance(value, np.generic):
            return plain(value.item())
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return '+Infinity' if value == math.inf else '-Infinity' if value == -math.inf else 'NaN'
        return value

    output = Path(args.output or HERE / (scenario['target'].replace(':', '_') + '_result.json'))
    output.parent.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    result = dict(scenario=scenario, scope='complete offline lower PICK preflight',
        observations_are_synthetic=True, live_freshness_validated=False,
        physical_retention_or_placement_validated=False,
        controller_calls=[], events=[], passed=False,
        harness_sha256=sha256(__file__), scenario_manifest_sha256=sha256(HERE / 'scenarios.json'),
        geometry_mode='serial' if args.serial else 'production owned pools',
        source_sha256={str(path.relative_to(package)): sha256(path)
                       for path in sorted((package / 'erc_phase1_solution').glob('*.py'))})
    stage = 'fixture'
    node = None
    timer = None

    def emit(event, **fields):
        nonlocal stage
        if fields.get('stage'):
            stage = fields['stage']
        record = dict(event=event, elapsed_s=time.monotonic() - began, **fields)
        result['events'].append(plain(record))
        print(json.dumps(plain(record), allow_nan=False), flush=True)

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    try:
        node = _official_manipulation_planner()
        del node._current_seed  # Restore ordinary measured-joint production seed.
        node.joints.update(zip(IK_JOINTS, HOME))
        node.joints['head_2_joint'] = scenario['head_pitch_rad']
        node.joints['gripper_left_finger_joint'] = .069
        node._lock = threading.Lock()
        node._cancel = threading.Event()
        # Logical replay time; these values are explicitly synthetic. Actual
        # admission functions still execute against this fixed stationary scene.
        node.get_clock = lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=1_000_000_000))
        node._joint_stamps_ns = {name: 1_000_000_000 for name in node.joints}
        node._staging_odom = dict(stamp_ns=1_000_000_000, pose=[0., 0., 0.],
                                  linear_speed=0., angular_speed=0.)
        node.gripper_open = .069
        node.adaptive_endpoint_tolerance = .0006
        node.carried_book_dimensions = np.array([.16, .02, .25])
        node.pick_torso_height = .35
        node.pregrasp_offset = .14
        node.pick_position_tolerance = .0005
        node.pick_orientation_tolerance = .01
        node.lift_first_extraction_lift_m = .020
        node._place_parallel_geometry_active = False
        node.empty_pickup_parallel_geometry_enabled = not args.serial
        node.pickup_parallel_geometry_enabled = not args.serial
        node.pickup_post_retreat_parallel_geometry_enabled = True
        node.geometry_process_workers = 8
        urdf = Path(get_package_share_directory('erc_description')) / 'urdf' / 'tiago_pro.urdf'
        node._shelf_cradle_geometry = ShelfCradleGeometry(
            urdf, get_package_share_directory, immutable_local=True)
        # Same official facets, with runtime immutable ownership enabled.
        meshes = []
        for mesh in node.carried_collision_meshes:
            owner = ModelLocalMesh(mesh.triangles)
            meshes.append(CollisionMesh(mesh.link, owner.snapshot()[0], mesh.bounds,
                                        mesh.watertight, owner))
        node.carried_collision_meshes = tuple(meshes)
        node._publish_status = lambda event, **fields: emit('status', status=event, **fields)

        def forbidden(*values, **options):
            result['controller_calls'].append(dict(args=repr(values), kwargs=repr(options)))
            raise AssertionError('offline coverage attempted controller dispatch')

        for name in ('_move_arm', '_move_left', '_move_torso', '_move_head',
                     '_move_gripper', '_follow', '_follow_timed_trajectory'):
            setattr(node, name, forbidden)
        initialize_pickup_geometry_backend(node, get_package_share_directory)
        identity = node._pickup_parallel_geometry_identity
        result['urdf_sha256'] = sha256(urdf)
        result['installed_geometry_identity'] = None if identity is None else dict(
            source_id=identity.source_id, model_id=identity.model_id)
        front = np.array(scenario['front_base_m'], dtype=float)
        bay = RelativeShelfBay(**scenario['bay'])
        marker, left, side_min, side_max = _bay_values(bay)
        require(abs(float((front - marker) @ left) - scenario['spawn_jitter_m']) < 1e-12,
                'synthetic_lateral_transform_does_not_preserve_spawn_offset')
        require(np.allclose([side_min, side_max], [-.215, .355], rtol=0., atol=1e-12),
                'physical_column_one_bounds_changed')
        guard = EmptyPickupCollision.capture(node)
        bounds = EmptyShelfBounds(node, front, None, guard, bay)
        require(bounds.normal_uncertainty_m == .010, 'normal_uncertainty_changed')
        result['initial_fixture'] = dict(left=guard.start, right=guard.right, head=guard.head,
            aperture=guard.initial_aperture, book_dimensions=node.carried_book_dimensions,
            transition_samples=node.carried_transition_samples, marker=marker, left_axis=left,
            effective_sides=[side_min, side_max], registered_lip=bounds.registered_lip,
            effective_plane=bounds.plane_point, book_depth_m=bounds.book_depth_m,
            floor_m=bounds.floor, roof_m=bounds.roof)
        reference = node._lift_first_measurements()

        def lift(grasp, extraction, *, post_retreat_plan=None):
            # Same runtime continuation and fresh-context checks. Empty workers
            # must already be closed before the loaded owner starts.
            require(not node._pickup_parallel_geometry_active, 'empty_pool_not_closed')
            require(not hasattr(guard, '_parallel_sequence'), 'empty_sequence_still_owned')
            node._lift_first_measurements(reference)
            return run_pickup_geometry(node, plan_lift_first_extraction,
                front, grasp, extraction, scene_reference=reference,
                post_retreat_plan=post_retreat_plan, bay=bay,
                aperture=float(node.carried_book_dimensions[1]),
                lift_m=node.lift_first_extraction_lift_m, modeled_tool_allowance_m=.005)

        # Cooperative cancellation preserves normal pool cleanup. An external
        # SIGINT may also be used; no detached or background process is created.
        timer = threading.Timer(args.wall_seconds, node._cancel.set)
        timer.daemon = True
        timer.start()
        stage = 'complete_lower_pick_admission'
        plan = helper.plan_lower_shelf_pick(node, front, empty_guard=guard,
                                          lift_planner=lift, bay=bay)
        cache = plan.cached_post_retreat_plan
        require(cache is not None and bool(cache['legs']), 'deferred_carry_cache_missing')
        require(np.array_equal(cache['start'], plan.lift_plan.terminal), 'carry_start_changed')
        require(node.pick_torso_height == .35, 'global_torso_configuration_changed')
        require(cache['compact_radius'] < node.carried_navigation_radius_limit,
                'compact_radius_outside_existing_limit')
        require(abs(cache['shelf_front_x'] - (front[0] + .25)) < 1e-12,
                'post_retreat_shelf_plane_changed')
        require(bool(plan.unloaded_recovery_route), 'unloaded_recovery_proposal_missing')
        require(not node._pickup_parallel_geometry_active and not hasattr(guard, '_parallel_sequence'),
                'geometry_owner_remained_active')
        guard.require_fresh(guard.start, guard.initial_aperture)
        node._lift_first_measurements(reference)
        if identity is not None:
            identity.verify()
        result['plan'] = dict(candidate=plan.candidate_name, positions=plan.positions,
            grasp=plan.grasp, rotations=plan.rotations, solutions=plan.solutions,
            transition=plan.transition_waypoints, extraction=plan.extraction_solutions,
            selected_torso=plan.pick_torso_height, loaded_index=plan.loaded_clearance_index,
            lift_route=plan.lift_plan.route, lift_metrics=dict(plan.lift_plan.metrics),
            carry=cache, return_result=plan.return_result,
            recovery_proposal=plan.unloaded_recovery_route,
            shelf_metrics=dict(plan.shelf_geometry_metrics))
        result['empty_state'] = dict(samples=guard.checked_samples, cache_hits=guard.cache_hits,
                                     last_rejection=guard.last_rejection)
        stage = 'retained_support_audit'
        previous = cache['start']
        for goal, phase in cache['legs']:
            if phase not in ('post_retreat_clearance_extension', 'post_retreat_cradle_roll'):
                require(node._gravity_supported_transition_is_safe(previous, goal),
                        'ordinary_supported_carry_predicate_rejected:' + phase)
            previous = goal
        # Restore ordinary measured-state access with a separately labeled
        # synthetic carried endpoint for the usual look-bin head preflight.
        stage = 'look_bin_head_preflight'
        node._held_book_corners = cache['attached_corners']
        node.joints.update(zip(IK_JOINTS, cache['terminal']))
        require(node._carried_head_transition_is_safe(0., -.60), 'look_bin_head_sweep_rejected')
        require(not node._cancel.is_set(), 'coverage_wall_budget_or_cancellation')
        require(not result['controller_calls'], 'controller_calls_recorded')
        result.update(passed=True,
            complete_empty_lift_extraction_cached_carry_body_tool_bay_admission=True,
            look_bin_head_preflight=True,
            recovery_scope='ordinary unloaded proposal only; no lower retained-failure motion certificate')
        emit('OFFLINE_SYNTHETIC_COVERAGE_PASS', target=scenario['target'],
             candidate=plan.candidate_name, radius=cache['compact_radius'])
    except BaseException as error:
        result.update(failure_stage=stage, error_type=type(error).__name__, error=str(error))
        emit('OFFLINE_SYNTHETIC_COVERAGE_FAIL', target=scenario['target'],
             failure_stage=stage, error=repr(error))
        raise
    finally:
        if timer is not None:
            timer.cancel()
        result['elapsed_s'] = time.monotonic() - began
        result['geometry_owner_active_on_exit'] = (
            None if node is None else getattr(node, '_pickup_parallel_geometry_active', False))
        output.write_text(json.dumps(plain(result), indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe', action='store_true')
    mode.add_argument('--run', choices=('3:yellow', '3:green'))
    parser.add_argument('--package', default='/opt/erc_ws/src/erc_phase1_solution')
    parser.add_argument('--output')
    parser.add_argument('--serial', action='store_true', help='Same checks with existing serial geometry backend')
    parser.add_argument('--wall-seconds', type=float, default=300.)
    args = parser.parse_args()
    if not math.isfinite(args.wall_seconds) or not 1. <= args.wall_seconds <= 600.:
        parser.error('--wall-seconds must be from 1 through 600')
    specification = json.loads((HERE / 'scenarios.json').read_text())
    if args.describe:
        print(json.dumps(specification, indent=2))
    else:
        run(args, next(item for item in specification['scenarios'] if item['target'] == args.run))


if __name__ == '__main__':
    main()
