"""DRAFT bottom PLACE: negative wrist plus registered anchor proposals.

Only caller proposal admission and pure solver anchor activation differ.
All body/tool/payload/scene/opening/static-release predicates remain production.
This is a draft algorithm probe, not production or physical mission evidence.
"""
import argparse
from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace as NS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('row', choices=('bottom',))
    parser.add_argument('--workspace', type=Path, default=Path('/opt/erc_ws'))
    parser.add_argument('--cap-seconds', type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.cap_seconds <= 300:
        parser.error('external cap must be between 1 and 300 seconds')
    root = args.workspace.resolve()
    package = root / 'src/erc_phase1_solution'
    sys.path[:0] = [str(package), str(package / 'test')]
    import numpy as np
    import yaml
    from ament_index_python.packages import get_package_share_directory
    from test_shutdown import _official_manipulation_planner
    import importlib.util
    import erc_phase1_solution as package_module
    draft_root = Path(__file__).resolve().parent
    for name in ('bottom_place_proposals', 'scene_cartesian_solver', 'scene_checked_place'):
        full = 'erc_phase1_solution.' + name
        spec = importlib.util.spec_from_file_location(full, draft_root / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
        setattr(package_module, name, module)
    planner = sys.modules['erc_phase1_solution.scene_checked_place']
    from erc_phase1_solution.adaptive_grasp import GripperFeedback
    from erc_phase1_solution.book_centered_place import book_centered_place_target
    from erc_phase1_solution.motion_profiles import IK_JOINTS
    from erc_phase1_solution.place_contact_guard import PlaceContactGuard
    from erc_phase1_solution.placement_scene_context import measured_scene_context
    from erc_phase1_solution.release_only_place_planning import capture, require_plan
    from erc_phase1_solution.shelf_cradle_geometry import ShelfCradleGeometry
    from erc_phase1_solution.exact_local_bounds import ModelLocalMesh
    from erc_phase1_solution.kinematics import CollisionMesh

    def plain(value):
        if isinstance(value, np.ndarray):
            return plain(value.tolist())
        if isinstance(value, np.generic):
            return plain(value.item())
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, float) and not np.isfinite(value):
            return '+Infinity' if value == np.inf else '-Infinity' if value == -np.inf else 'NaN'
        return value

    here = Path(__file__).resolve().parent
    fixture_path = here / (args.row + '_carry.json')
    scene_path = here / 'recorded_place_inputs.json'
    fixture = json.loads(fixture_path.read_text())
    recorded = json.loads(scene_path.read_text())
    profile_path = package / 'config/collision_quality.yaml'
    profile = yaml.safe_load(profile_path.read_text())['erc_manipulation']['ros__parameters']
    source_hashes = {str(path.relative_to(package)): digest(path)
                     for path in sorted((package / 'erc_phase1_solution').glob('*.py'))}
    result = dict(row=args.row, passed=False, exact_live_replay=False,
        scope='DRAFT bottom-only registered anchors plus nearest feasible negative wrist; synthetic bottom carry + recorded scene',
        draft_sources={name: digest(draft_root / name) for name in ('bottom_place_proposals.py', 'scene_cartesian_solver.py', 'scene_checked_place.py')},
        current_production_proof=False,
        live_freshness_or_retention_or_delivery_validated=False,
        source_sha256=source_hashes, profile_sha256=digest(profile_path),
        harness_sha256=digest(__file__), carry_fixture_sha256=digest(fixture_path),
        recorded_place_inputs_sha256=digest(scene_path), carry_provenance=fixture['provenance'],
        controller_calls=[], worker_processes_created_by_harness=0,
        assumptions=[
            'Row-specific carry, staging and attachment are saved offline complete PICK outputs.',
            'Exact selected bin/table/base/right/head and master aperture are from trial105d91426ddd blue PLACE.',
            'Combining these inputs is synthetic; those lower books have not been observed at this bin pose.',
            'Logical fresh clock/feedback stamps are fixture values; all ordinary admission methods execute.',
            'Original measured scene coordinates and registration margins are retained verbatim.',
            'Serial production geometry predicates run; no worker-pool performance claim.',
            'Ordinary default negative wrist policy and release-only static endpoint checks remain active.'])
    started = time.monotonic()
    stage = 'fixture'
    events = []
    node = None

    def emit(event, **fields):
        nonlocal stage
        stage = fields.get('stage', stage)
        row = dict(event=event, elapsed_seconds=time.monotonic() - started, **fields)
        events.append(plain(row))
        print(json.dumps(plain(row), allow_nan=False), flush=True)

    def require(condition, reason):
        if not condition:
            raise RuntimeError(reason)

    try:
        expected_module = draft_root / 'scene_checked_place.py'
        require(Path(planner.__file__).resolve() == expected_module.resolve(),
                'replay_requires_the_declared_draft_module')
        node = _official_manipulation_planner()
        del node._current_seed
        node._lock = threading.RLock()
        node._cancel = threading.Event()

        def external_timeout(*unused):
            node._cancel.set()
            raise TimeoutError('external offline PLACE geometry cap')

        signal.signal(signal.SIGALRM, external_timeout)
        signal.alarm(args.cap_seconds)
        node.carried_book_dimensions = np.array([.16, .02, .25])
        require(np.array_equal(node.carried_book_dimensions, fixture['book_dimensions_m']),
                'saved_carry_dimensions_differ_from_official_book')
        reconstructed = node._attached_book_corners(fixture['front'], fixture['grasp_solution'])
        saved = np.array(fixture['attached_corners'], dtype=float)
        result['attachment_recompute_max_abs_difference'] = float(np.max(np.abs(reconstructed - saved)))
        require(np.array_equal(reconstructed, saved), 'saved_attachment_does_not_exactly_match_current_production')
        node._held_book_corners = reconstructed
        # Cache metadata describes its initial vertical-pinch entry. The
        # ordinary executor enables supported mode during the cradle roll;
        # its final compact terminal is the state supplied to PLACE here.
        require(fixture['requires_gravity_support'] is False,
                'saved_deferred_carry_initial_mode_changed')
        carried = np.array(fixture['carry_terminal'], dtype=float)
        require(carried.shape == (8,) and carried[-1] < 0., 'fixture_does_not_use_default_negative_wrist')
        require(node._gravity_supported_transition_is_safe(carried, carried),
                'saved_final_carry_terminal_fails_ordinary_support_predicate')
        node._gravity_supported_payload = True
        node._transport_lock_engaged = True
        node._payload_monitor_enabled = True
        result['synthetic_retained_state_assumption'] = dict(transport_lock=True,
            payload_monitor=True, gravity_supported=True, live_bottom_contact_proof=False)
        place_height = float(profile['place_torso_height'])
        require(place_height == .35, 'profile_PLACE_torso_changed_from_reviewed_input')
        node.place_torso_height = place_height
        node.scene_cartesian_wall_seconds = float(profile['scene_cartesian_wall_seconds'])
        torso_ready = carried.copy(); torso_ready[0] = place_height
        staging = np.array(fixture['staging_terminal'], dtype=float); staging[0] = place_height
        node._carried_staging_solution = staging.copy()
        node.gripper_open = .069
        node.joints.update(zip(IK_JOINTS, carried))
        node.joints.update(recorded['selected_admission_reference']['parked_joints'])
        node.joints['gripper_left_finger_joint'] = recorded['measured_master']
        node._joint_stamps_ns = {name: 1_000_000_000 for name in node.joints}
        node.get_clock = lambda: NS(now=lambda: NS(nanoseconds=1_000_000_000))
        reference = recorded['selected_admission_reference']
        node._staging_odom = dict(stamp_ns=1_000_000_000, producer_stamp_ns=1_000_000_000,
            pose=reference['base_pose'], linear_speed=0., angular_speed=0.,
            frame_id=reference['odom_frame_id'], child_frame_id=reference['odom_child_frame_id'])
        node._gripper_feedback_samples = deque([
            GripperFeedback(1_000_000_000, recorded['measured_master'], 0.)])
        urdf = Path(get_package_share_directory('erc_description')) / 'urdf/tiago_pro.urdf'
        node._shelf_cradle_geometry = ShelfCradleGeometry(
            urdf, get_package_share_directory, immutable_local=True)
        meshes = []
        for mesh in node.carried_collision_meshes:
            owner = ModelLocalMesh(mesh.triangles)
            meshes.append(CollisionMesh(mesh.link, owner.snapshot()[0], mesh.bounds,
                                        mesh.watertight, owner))
        node.carried_collision_meshes = tuple(meshes)
        node._publish_status = lambda event, **fields: emit(event, **fields)

        def forbidden(*values, **options):
            result['controller_calls'].append(dict(args=repr(values), kwargs=repr(options)))
            raise AssertionError('offline geometry attempted controller dispatch')

        for name in ('_move_arm', '_move_torso', '_move_head', '_move_gripper', '_follow'):
            setattr(node, name, forbidden)
        identity = dict(trial_id='synthetic_lower_place_' + args.row,
            placement_attempt_id='offline_' + args.row + '_recorded105d91426ddd',
            target_model=fixture['target_model_for_identity'])
        payload = dict(identity, completion_mode='release_pose')
        node._busy = True
        node._target_book_model = identity['target_model']
        node._goal_handles = []
        node._pending_retained_acceptances = []
        node._release_pose_owner = None
        for name in ('release_only_place_planning_enabled', 'place_finish_at_release_enabled',
                     'delivery_evidence_enabled', 'book_centered_place_enabled',
                     'table_scene_required', 'bin_scene_required'):
            require(profile[name] is True, 'profile_release_policy_changed:' + name)
            setattr(node, name, True)
        node.bin_clearance_timing_enabled = False
        node._selected_place_scene_reference = measured_scene_context(node)
        node._active_place_scene_reference = node._selected_place_scene_reference
        node._selected_place_bin_scene = deepcopy(recorded['selected_bin_scene'])
        node._selected_place_table_scene = deepcopy(recorded['selected_table_scene'])
        node._place_contact_guard = PlaceContactGuard(1_000_000_000, identity)
        request = capture(node, payload)
        floor = np.array(recorded['bin_floor_point'], dtype=float)
        target = book_centered_place_target(reconstructed, floor,
            float(profile['book_centered_place_height_above_point_m']),
            bin_rotation=recorded['selected_bin_scene']['rotation'])
        release = target.tool_position.copy()
        above = release.copy()
        above[2] += max(.13, float(profile['book_centered_place_clearance_height_m'])
                        - float(profile['book_centered_place_height_above_point_m']))
        clearance = above.copy(); clearance[0] = float(profile['book_centered_place_approach_x_m'])
        positions = [clearance,
            *node._interpolate_positions(clearance, above, node.cartesian_step),
            *node._interpolate_positions(above, release, node.cartesian_step)]
        result['inputs'] = dict(front=fixture['front'], grasp=fixture['grasp_solution'],
            attached=reconstructed, carried_start=carried, torso_ready=torso_ready,
            staging=staging, positions=positions, rotation=target.tool_rotation,
            floor=floor, master=recorded['measured_master'], identity=identity,
            selected_scene_reference=node._selected_place_scene_reference,
            selected_bin_scene=node._selected_place_bin_scene,
            selected_table_scene=node._selected_place_table_scene)
        emit('INPUTS_ADMITTED', row=args.row, wrist=carried[-1],
             attachment_difference=result['attachment_recompute_max_abs_difference'])
        stage = 'complete_scene_checked_PLACE'
        planning_started = time.monotonic()
        plan = planner.plan_scene_checked_place(node, positions, target.tool_rotation,
            carried, torso_ready, staging, floor, get_package_share_directory,
            table_scene=node._selected_place_table_scene,
            bin_scene=node._selected_place_bin_scene, release_only_request=request)
        stage = 'release_only_endpoint_identity'
        endpoint = require_plan(request, plan, node, identity)
        require(endpoint is plan.release_only_endpoint, 'returned_endpoint_identity_changed')
        require(all(np.asarray(q)[-1] < 0. for q in plan.solutions),
                'draft_negative_wrist_rule_not_preserved')
        selected_position = target.tool_position if plan.selected_target is None else plan.selected_target.tool_position
        selected_diagnostics = plan.diagnostics.get('registered_target_proposal')
        if selected_diagnostics is not None:
            require(selected_diagnostics['selected_minimum_wall_reserve_m'] >=
                    selected_diagnostics['original_minimum_wall_reserve_m'] - 1e-12,
                    'selected_proposal_reduced_original_wall_reserve')
        require(not result['controller_calls'], 'controller_commands_recorded')
        result.update(passed=True, planning_seconds=time.monotonic() - planning_started,
            plan=dict(solutions=plan.solutions, setup=plan.setup, diagnostics=plan.diagnostics,
                release_endpoint=endpoint.goal, selected_target_position=selected_position,
                registered_target_override=selected_diagnostics),
            complete_static_release_only_endpoint_checked=True)
        emit('DRAFT_NEGATIVE_REGISTERED_PLACE_PASS', row=args.row,
             planning_seconds=result['planning_seconds'], diagnostics=plan.diagnostics)
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), failure_stage=stage,
                      failure_diagnostics=deepcopy(getattr(error, 'diagnostics', None)))
        emit('DRAFT_NEGATIVE_REGISTERED_PLACE_FAIL', row=args.row, failure_stage=stage,
             error_type=type(error).__name__, error=str(error),
             diagnostics=result['failure_diagnostics'])
    finally:
        signal.alarm(0)
        result.update(total_seconds=time.monotonic() - started, events=events)
        result['source_unchanged_through_run'] = all(
            digest(package / name) == before for name, before in source_hashes.items())
        (here / (args.row + '_place_result.json')).write_text(
            json.dumps(plain(result), indent=2, allow_nan=False) + '\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
