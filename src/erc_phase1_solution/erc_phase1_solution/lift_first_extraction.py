"""Optional, bounded lift-first planning; no ROS nodes or motion dispatch.

Bay dimensions come from the unchanged official b1f9b05 shelf mesh. The caller
registers them with onboard marker geometry. Initial upright book support is an
explicit diagnostic assumption, not a measured shelf-floor collision model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping, Sequence

import numpy as np

from .kinematics import pose_matrix
from .rigid_palm_preflight import FINGER_COLLISION_LINKS, PALM_COLLISION_LINK
from .shelf_cradle_geometry import check_cradle_tool_sweep


# Inner shelf walls relative to the corresponding marker centre, increasing
# toward the robot's left. Mesh column pitch is not exactly marker pitch.
_COLUMN_SIDES = ((-.430, .570), (-.465, .535), (-.500, .500),
                 (-.535, .465), (-.570, .430))
_BAY_HEIGHT = .300


@dataclass(frozen=True)
class RelativeShelfBay:
    """Onboard registration plus disclosed uncertainty for a local diagnostic.

    ``inward_axis_base`` points horizontally into the shelf. Marker XYZ must
    come from onboard RGB-D/TF, with column assigned by visual registration.
    The caller must bound accumulated registration/TF error in these supplied
    uncertainties and use current base-frame coordinates. They are not inferred
    from a digit classifier's confidence or from evaluator/simulator poses.
    """
    marker_center_base: Sequence[float]
    inward_axis_base: Sequence[float]
    physical_column: int
    lateral_uncertainty_m: float
    roof_uncertainty_m: float
    assume_upright_supported: bool
    source: str
    margin_m: float = .015


@dataclass(frozen=True)
class LiftFirstPlan:
    route: tuple[np.ndarray, ...]
    terminal: np.ndarray
    attached_corners: np.ndarray
    metrics: Mapping[str, object]


def _finite_vector(value, size, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{label} must contain {size} finite values')
    return result.copy()


def _check_cancelled(node):
    cancelled = getattr(node, '_cancel', None)
    if cancelled is not None and cancelled.is_set():
        raise ValueError('lift-first geometry check cancelled')


def _pose(chain, joints):
    pose = np.asarray(chain.forward(joints), dtype=float)
    if (pose.shape != (4, 4) or not np.all(np.isfinite(pose))
            or not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-9)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), rtol=0, atol=1e-7)
            or abs(np.linalg.det(pose[:3, :3])-1) > 1e-7):
        raise ValueError('lift-first FK must be a finite rigid transform')
    return pose


def _bay_values(bay):
    if not isinstance(bay, RelativeShelfBay) or bay.assume_upright_supported is not True:
        raise ValueError('lift-first requires explicit upright supported-book assumption')
    if not isinstance(bay.source, str) or not bay.source.strip():
        raise ValueError('lift-first requires onboard registration provenance')
    if (isinstance(bay.physical_column, bool)
            or int(bay.physical_column) != bay.physical_column
            or not 1 <= bay.physical_column <= 5):
        raise ValueError('physical shelf column must be from 1 through 5')
    marker = _finite_vector(bay.marker_center_base, 3, 'marker centre')
    inward = _finite_vector(bay.inward_axis_base, 3, 'shelf inward axis')
    if abs(inward[2]) > 1e-6 or abs(np.linalg.norm(inward)-1.) > 1e-6:
        raise ValueError('shelf inward axis must be horizontal and unit length')
    limits = (bay.lateral_uncertainty_m, bay.roof_uncertainty_m, bay.margin_m)
    if not all(math.isfinite(x) and x >= 0 for x in limits):
        raise ValueError('bay uncertainties and margin must be finite and nonnegative')
    if bay.margin_m < .015:
        raise ValueError('lift-first preserves at least 15 mm roof/side margin')
    left = np.asarray([-inward[1], inward[0], 0.])
    lower, upper = _COLUMN_SIDES[int(bay.physical_column)-1]
    allowance = bay.lateral_uncertainty_m + bay.margin_m
    if lower + allowance >= upper - allowance:
        raise ValueError('bay lateral uncertainty leaves no usable opening')
    return marker, left, lower+allowance, upper-allowance


def plan_lift_first_extraction(
    node,
    front: Sequence[float],
    grasp_solution: Sequence[float],
    original_loaded_goals: Sequence[Sequence[float]],
    *,
    bay: RelativeShelfBay,
    aperture: float,
    lift_m: float = .005,
    finger_positions: Mapping[str, float] | None = None,
    modeled_tool_allowance_m: float = 0.,
    geometry_backend=None,
) -> LiftFirstPlan:
    """Solve a rise followed by raised copies of the loaded withdrawal goals.

    No input approach array is changed and no cached carry plan is updated.
    ``modeled_tool_allowance_m`` is an engineering roof/side allowance only,
    not a force-derived bound on passive hinge deflection.
    Integration must regenerate ``_plan_carried_return`` from ``terminal`` with
    the ORIGINAL front/grasp transform, and preflight its recovery/cache before
    dispatch. Live node collision methods supply measured right/head state.
    """
    marker, left, side_min, side_max = _bay_values(bay)
    if not math.isfinite(modeled_tool_allowance_m) or not 0 <= modeled_tool_allowance_m <= .005:
        raise ValueError('modeled tool allowance must be finite and within 5 mm')
    if not math.isfinite(lift_m) or not 0 < lift_m <= .020:
        raise ValueError('lift-first rise must be positive and at most 20 mm')
    if not math.isfinite(aperture) or not 0 <= aperture <= .069:
        raise ValueError('lift-first aperture is outside the gripper range')
    chain = node.chain
    count = len(chain.active_names)
    if count != 8:
        raise ValueError('lift-first expects the official eight-joint arm/torso chain')
    point = _finite_vector(front, 3, 'book front')
    grasp = _finite_vector(grasp_solution, count, 'grasp solution')
    original = [_finite_vector(q, count, 'withdrawal goal') for q in original_loaded_goals]
    if not original:
        raise ValueError('lift-first requires an existing loaded withdrawal route')
    for q in (grasp, *original):
        if np.any(q < chain.lower) or np.any(q > chain.upper):
            raise ValueError('lift-first input joint limit violation')
    dimensions = _finite_vector(node.carried_book_dimensions, 3, 'book dimensions')
    if not np.allclose(dimensions, [.16, .02, .25], rtol=0, atol=1e-9):
        raise ValueError('lift-first requires the current official 20 mm book geometry')
    pos_tol = float(getattr(node, 'pick_position_tolerance', .0005))
    rot_tol = float(getattr(node, 'pick_orientation_tolerance', .01))
    if not (0 < pos_tol <= .0005 and 0 < rot_tol <= .01):
        raise ValueError('lift-first requires the existing precise pick tolerances')
    if lift_m <= 2*pos_tol:
        raise ValueError('lift-first rise must exceed the position-error allowance')
    minimum_withdrawal_rise = lift_m - 2*pos_tol
    geometry = getattr(node, '_shelf_cradle_geometry', None)
    if geometry is None:
        raise ValueError('lift-first requires the existing nine-link gripper geometry')
    surfaces = geometry.local_surfaces(aperture, finger_positions)
    required = {PALM_COLLISION_LINK, *FINGER_COLLISION_LINKS}
    if not required.issubset(surfaces):
        raise ValueError('lift-first gripper collision surfaces are incomplete')
    tool = np.concatenate([np.asarray(surfaces[name], dtype=float).reshape(-1, 3)
                           for name in sorted(required)])
    if not len(tool) or not np.all(np.isfinite(tool)):
        raise ValueError('lift-first gripper collision surfaces are invalid')
    initial = _pose(chain, grasp)
    rotation = initial[:3, :3].copy()
    signs = np.asarray([(x, y, z) for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    initial_book = point + [dimensions[0]/2, 0., 0.] + signs * (dimensions/2)
    local_book = (initial_book-initial[:3, 3]) @ rotation
    attached = np.asarray(node._attached_book_corners(point, grasp), dtype=float).copy()
    if attached.shape != (8, 3) or not np.all(np.isfinite(attached)):
        raise ValueError('lift-first attached payload envelope is invalid')
    floor = float(initial_book[:, 2].min())
    roof = floor + _BAY_HEIGHT - bay.roof_uncertainty_m - bay.margin_m
    offset = np.asarray([0., 0., lift_m])
    desired = [initial[:3, 3] + offset]
    desired.extend(_pose(chain, q)[:3, 3] + offset for q in original)
    route, records, previous = [], [], grasp.copy()
    for index, position in enumerate(desired):
        target = pose_matrix(position, rotation)
        seeds = [previous.copy()]
        if index:
            seeds.append(original[index-1].copy())
        solved, _ = chain.solve(target, seeds, position_tolerance=pos_tol,
            orientation_tolerance=rot_tol,
            fixed_positions={'torso_lift_joint': float(grasp[0])})
        if solved is None:
            raise ValueError(f'lift-first IK failed at leg {index}')
        solved = _finite_vector(solved, count, 'lift-first IK solution')
        actual = _pose(chain, solved)
        error = chain.pose_error(actual, target)
        if (np.linalg.norm(error[:3]) > pos_tol or np.linalg.norm(error[3:]) > rot_tol
                or np.any(solved < chain.lower) or np.any(solved > chain.upper)
                or abs(solved[0]-grasp[0]) > 1e-9):
            raise ValueError(f'lift-first IK residual or joint limit at leg {index}')
        volume_check = (node._carried_robot_transition_is_safe if geometry_backend is None
                        else geometry_backend.volume)
        if not volume_check(previous, solved, attached):
            raise ValueError(f'lift-first robot/payload sweep rejected leg {index}')
        tool_reason = (check_cradle_tool_sweep(node, point, grasp, previous, solved,
            None, aperture=aperture, finger_positions=finger_positions)
            if geometry_backend is None else geometry_backend.tool_sweep(
                point, grasp, previous, solved, None,
                aperture=aperture, finger_positions=finger_positions))
        if tool_reason is not None:
            raise ValueError(f'lift-first gripper sweep rejected leg {index}: {tool_reason}')
        samples = max(61, int(node.carried_transition_samples),
            int(math.ceil(np.max(np.abs(solved[1:]-previous[1:]))/.02))+1)
        minimum_rise, minimum_roof, minimum_side = math.inf, math.inf, math.inf
        for fraction in np.linspace(0., 1., samples):
            transform = _pose(chain, previous + (solved-previous)*fraction)
            book = local_book @ transform[:3, :3].T + transform[:3, 3]
            rise = float(np.min(book[:, 2]-initial_book[:, 2]))
            if rise < -1e-9:
                raise ValueError(f'lift-first book-corner descent at leg {index}')
            if index > 0 and rise < minimum_withdrawal_rise:
                raise ValueError(f'lift-first maintained floor rise at leg {index}: '
                    f'observed_m={rise!r}, required_m={minimum_withdrawal_rise!r}')
            tool_points = tool @ transform[:3, :3].T + transform[:3, 3]
            # Extra allowance applies only to inferred finger/tool geometry.
            # Book roof/side uncertainty and margins retain their prior values.
            roof_clearance = float(min(roof-book[:, 2].max(),
                roof-tool_points[:, 2].max()-modeled_tool_allowance_m))
            lateral = (book-marker) @ left
            tool_lateral = (tool_points-marker) @ left
            side_clearance = float(min(lateral.min()-side_min, side_max-lateral.max(),
                tool_lateral.min()-side_min-modeled_tool_allowance_m,
                side_max-tool_lateral.max()-modeled_tool_allowance_m))
            if roof_clearance < 0:
                raise ValueError(f'lift-first uncertain bay roof clearance at leg {index}')
            if side_clearance < 0:
                raise ValueError(f'lift-first uncertain bay side clearance at leg {index}')
            minimum_rise = min(minimum_rise, rise)
            minimum_roof = min(minimum_roof, roof_clearance)
            minimum_side = min(minimum_side, side_clearance)
        if index == 0 and float(np.min(book[:, 2]-initial_book[:, 2])) < lift_m-2*pos_tol:
            raise ValueError('lift-first first endpoint does not provide the requested rise')
        route.append(solved.copy())
        records.append(dict(leg=index, samples=samples,
            position_error_m=float(np.linalg.norm(error[:3])),
            orientation_error_rad=float(np.linalg.norm(error[3:])),
            minimum_corner_rise_m=minimum_rise,
            required_minimum_corner_rise_m=(0. if index == 0 else minimum_withdrawal_rise),
            corner_rise_reserve_m=minimum_rise-(0. if index == 0 else minimum_withdrawal_rise),
            roof_clearance_after_margin_uncertainty_m=minimum_roof,
            side_clearance_after_margin_uncertainty_m=minimum_side))
        previous = solved
    return LiftFirstPlan(tuple(route), route[-1].copy(), attached, dict(
        lift_m=lift_m, legs=records, registration_source=bay.source,
        minimum_withdrawal_corner_rise_m=minimum_withdrawal_rise,
        maintained_floor_reference='initial_upright_book_corners',
        lateral_uncertainty_m=bay.lateral_uncertainty_m,
        roof_uncertainty_m=bay.roof_uncertainty_m, margin_m=bay.margin_m,
        modeled_tool_allowance_m=modeled_tool_allowance_m,
        modeled_tool_allowance_is_certified_deflection_bound=False,
        initial_upright_support_assumed=True, floor_reference_base_z_m=floor,
        known_bay_height_m=_BAY_HEIGHT,
        full_shelf_collision_certificate=False, deferred_carry_regenerated=False,
        scope='Relative book no-descent, maintained post-lift rise and payload/tool roof/side bounds; '
              'no shelf-floor/front-edge mesh certificate or motion/retention proof'))


def validate_lift_first_route(
    node, front, measured_start, route, *, bay, aperture, lift_m=.005,
    finger_positions=None, attached_corners=None, modeled_tool_allowance_m=0.,
    right_positions=None, head_positions=None,
    geometry_backend=None,
):
    """Recheck exact planned legs from measured contact pose, without solving IK.

    The upright book still rests at the perceived front at this first-contact
    check. Its no-descent reference is the actual starting hand transform. The
    preplanned padded attachment can be supplied for the robot sweep so its
    existing carry/recovery cache is not silently redefined. A supplied master
    position alone still uses unmeasured official mimic FK for passive joints;
    supplied joint positions are never automatically described as all measured.
    """
    _check_cancelled(node)
    context = {}
    for name, supplied, size in (
        ('right_positions', right_positions, 7),
        ('head_positions', head_positions, 2),
    ):
        if supplied is not None:
            # One private, immutable sensor snapshot for all sampled passes.
            # Omission retains the existing live-state behavior of other callers.
            values = _finite_vector(supplied, size, name)
            context[name] = np.frombuffer(values.tobytes(), dtype=float)
    validation_started = time.monotonic()
    publish = getattr(node, '_publish_status', None)

    def report_pass(index, phase, began):
        elapsed = time.monotonic() - began
        if callable(publish):
            publish('lift_first_geometry_progress', command='pick', leg=index,
                    geometry_pass=phase, pass_wall_seconds=elapsed,
                    validation_wall_seconds=time.monotonic()-validation_started)
        return elapsed

    marker, left, side_min, side_max = _bay_values(bay)
    if not math.isfinite(modeled_tool_allowance_m) or not 0 <= modeled_tool_allowance_m <= .005:
        raise ValueError('modeled tool allowance must be finite and within 5 mm')
    first = _finite_vector(measured_start, 8, 'measured lift start')
    point = _finite_vector(front, 3, 'book front')
    goals = [_finite_vector(q, 8, 'checked lift goal') for q in route]
    if not goals or not math.isfinite(lift_m) or not 0 < lift_m <= .020:
        raise ValueError('checked lift route or rise is invalid')
    pos_tol = float(getattr(node, 'pick_position_tolerance', .0005))
    if (not math.isfinite(pos_tol) or not 0 < pos_tol <= .0005
            or lift_m <= 2*pos_tol):
        raise ValueError('checked lift requires a finite positive floor reserve')
    if not math.isfinite(aperture) or not 0 <= aperture <= .069:
        raise ValueError('checked lift aperture is invalid')
    dimensions = _finite_vector(node.carried_book_dimensions, 3, 'book dimensions')
    if not np.allclose(dimensions, [.16, .02, .25], rtol=0, atol=1e-9):
        raise ValueError('checked lift requires current official 20 mm book geometry')
    minimum_withdrawal_rise = lift_m - 2*pos_tol
    geometry = getattr(node, '_shelf_cradle_geometry', None)
    if geometry is None:
        raise ValueError('checked lift requires complete gripper geometry')
    surfaces = geometry.local_surfaces(aperture, finger_positions)
    required = {PALM_COLLISION_LINK, *FINGER_COLLISION_LINKS}
    if not required.issubset(surfaces):
        raise ValueError('checked lift gripper geometry is incomplete')
    tool = np.concatenate([np.asarray(surfaces[name], dtype=float).reshape(-1, 3)
                           for name in sorted(required)])
    if not len(tool) or not np.all(np.isfinite(tool)):
        raise ValueError('checked lift gripper surfaces are invalid')
    initial = _pose(node.chain, first)
    signs = np.asarray([(x, y, z) for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    book_initial = point + [dimensions[0]/2, 0., 0.] + signs*(dimensions/2)
    local_book = (book_initial-initial[:3, 3]) @ initial[:3, :3]
    attached = np.asarray(attached_corners if attached_corners is not None else
        node._attached_book_corners(point, first), dtype=float)
    if attached.shape != (8, 3) or not np.all(np.isfinite(attached)):
        raise ValueError('checked lift attachment is invalid')
    roof = float(book_initial[:, 2].min()) + _BAY_HEIGHT - bay.roof_uncertainty_m - bay.margin_m
    previous, records = first.copy(), []
    for index, goal in enumerate(goals):
        _check_cancelled(node)
        leg_started = time.monotonic()
        if (np.any(goal < node.chain.lower) or np.any(goal > node.chain.upper)
                or np.any(previous < node.chain.lower) or np.any(previous > node.chain.upper)):
            raise ValueError('checked lift joint limit violation')
        volume_check = (node._carried_robot_transition_is_safe if geometry_backend is None
                        else geometry_backend.volume)
        if not volume_check(previous, goal, attached, **context):
            _check_cancelled(node)
            raise ValueError(f'checked lift robot/payload sweep rejected leg {index}')
        _check_cancelled(node)
        payload_seconds = report_pass(index, 'robot_payload', leg_started)
        tool_started = time.monotonic()
        reason = (check_cradle_tool_sweep(node, point, first, previous, goal, None,
            aperture=aperture, finger_positions=finger_positions, **context)
            if geometry_backend is None else geometry_backend.tool_sweep(
                point, first, previous, goal, None,
                aperture=aperture, finger_positions=finger_positions, **context))
        if reason is not None:
            _check_cancelled(node)
            raise ValueError(f'checked lift gripper sweep rejected leg {index}: {reason}')
        _check_cancelled(node)
        tool_seconds = report_pass(index, 'tool', tool_started)
        bay_started = time.monotonic()
        samples = max(61, int(node.carried_transition_samples),
            int(math.ceil(np.max(np.abs(goal[1:]-previous[1:]))/.02))+1)
        minimum_rise, minimum_roof, minimum_side = math.inf, math.inf, math.inf
        for fraction in np.linspace(0., 1., samples):
            _check_cancelled(node)
            transform = _pose(node.chain, previous+(goal-previous)*fraction)
            book = local_book @ transform[:3, :3].T + transform[:3, 3]
            rise = float(np.min(book[:, 2]-book_initial[:, 2]))
            if rise < -1e-9:
                raise ValueError(f'checked lift book-corner descent at leg {index}')
            if index > 0 and rise < minimum_withdrawal_rise:
                raise ValueError(f'checked lift maintained floor rise at leg {index}: '
                    f'observed_m={rise!r}, required_m={minimum_withdrawal_rise!r}')
            tool_points = tool @ transform[:3, :3].T + transform[:3, 3]
            # Extra allowance applies only to inferred finger/tool geometry.
            # Book roof/side uncertainty and margins retain their prior values.
            roof_clearance = float(min(roof-book[:, 2].max(),
                roof-tool_points[:, 2].max()-modeled_tool_allowance_m))
            lateral = (book-marker) @ left
            tool_lateral = (tool_points-marker) @ left
            side_clearance = float(min(lateral.min()-side_min, side_max-lateral.max(),
                tool_lateral.min()-side_min-modeled_tool_allowance_m,
                side_max-tool_lateral.max()-modeled_tool_allowance_m))
            if roof_clearance < 0 or side_clearance < 0:
                raise ValueError(f'checked lift uncertain bay clearance at leg {index}')
            minimum_rise = min(minimum_rise, rise)
            minimum_roof = min(minimum_roof, roof_clearance)
            minimum_side = min(minimum_side, side_clearance)
        if index == 0 and rise < lift_m-2*float(getattr(node, 'pick_position_tolerance', .0005)):
            raise ValueError('checked lift endpoint rise is insufficient')
        records.append(dict(leg=index, samples=samples,
            minimum_corner_rise_m=minimum_rise,
            required_minimum_corner_rise_m=(0. if index == 0 else minimum_withdrawal_rise),
            corner_rise_reserve_m=minimum_rise-(0. if index == 0 else minimum_withdrawal_rise),
            roof_clearance_after_margin_uncertainty_m=minimum_roof,
            side_clearance_after_margin_uncertainty_m=minimum_side,
            pass_wall_seconds=dict(robot_payload=payload_seconds, tool=tool_seconds,
                bay=report_pass(index, 'relative_bay', bay_started)),
            leg_wall_seconds=time.monotonic()-leg_started))
        previous = goal
    _check_cancelled(node)
    return dict(legs=records, validation_wall_seconds=time.monotonic()-validation_started,
                minimum_withdrawal_corner_rise_m=minimum_withdrawal_rise,
                maintained_floor_reference='initial_upright_book_corners',
                supplied_finger_positions_checked=finger_positions is not None,
                modeled_tool_allowance_m=modeled_tool_allowance_m,
                modeled_tool_allowance_is_certified_deflection_bound=False,
                full_shelf_collision_certificate=False, route_replanned=False)
