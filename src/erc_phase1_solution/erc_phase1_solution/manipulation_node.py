"""Single-left-arm manipulation server with a cross-release gripper adapter."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
from pathlib import Path

from .arm_trajectory_timing import checked_arm_speed_scale, scaled_arm_seconds
from .compact_timing import checked_supported_compact_minimum_seconds
from .additional_arm_timing import checked_additional_arm_time_scale, retime_admitted_arm_goal, require_retimed_arm_headroom
from .geometry_process_pool import checked_geometry_process_workers
from .empty_pickup_setup_timing import (
    checked_empty_pickup_setup_retiming_enabled, move_empty_pickup_setup,
)
from .place_geometry_backend import (
    checked_place_parallel_geometry_enabled, initialize_place_geometry_backend,
    checked_place_carried_volume_parallel_enabled,
    plan_node_scene_checked_place,
)
from .empty_pickup_geometry_backend import (
    checked_empty_pickup_parallel_geometry_enabled, empty_pickup_geometry_scope,
)
from .pickup_geometry_backend import (
    checked_pickup_parallel_geometry_enabled, initialize_pickup_geometry_backend,
    run_pickup_geometry, checked_pickup_post_retreat_parallel_geometry_enabled,
)
from .withdrawal_timing import checked_withdrawal_speed_scale, withdrawal_leg_speed_scales
from .withdrawal_half_timing import (
    checked_enabled as checked_withdrawal_half_timing_enabled,
    normal_options as withdrawal_half_normal_options,
    validate_execution as validate_withdrawal_half_execution,
    timing_input as withdrawal_half_timing_input,
    require_publication_locked as require_withdrawal_half_publication_locked,
    require_endpoint as require_withdrawal_half_endpoint,
)
from .arm_velocity_admission import (
    ArmVelocityAdmissionRejected, require_arm_velocity_locked,
    publish_arm_velocity_admission,
)
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.clock import ClockType
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import JointState, PointCloud
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .adaptive_grasp import (
    AdaptiveGraspEvidence,
    ForceSample,
    GripperFeedback,
    evaluate_bilateral_contact,
)
from .common import (
    RELIABLE_QOS,
    SENSOR_QOS,
    TRANSIENT_RELIABLE_QOS,
    decode_event,
    encode_event,
    yaw_from_quaternion,
)
from .bin_geometry import bin_message_is_current, bin_message_is_verified
from .book_centered_place import book_centered_place_target
from .placement_scene_context import measured_scene_context, matching_table_scene
from . import place_transition_stop as _place_transition_stop
from . import release_pose_finish
from . import initial_stow_serial_timing as _initial_stow_serial
from . import release_only_place_planning
from . import release_only_clearance_timing as _release_clearance
from .bin_clearance_timing import (
    checked_enabled as checked_bin_clearance_timing_enabled,
    normal_options as bin_clearance_normal_options,
    validate_execution as validate_bin_clearance_execution,
    timing_input as bin_clearance_timing_input,
    require_endpoint as require_bin_clearance_endpoint,
    require_publication_locked as require_bin_clearance_publication_locked,
)
from .bin_scene_admission import match_registered_scenes
from .place_contact_guard import (
    activate as _activate_place_contacts, deactivate as _deactivate_place_contacts,
    observe as _observe_place_contacts, fault_reason as _place_contact_fault,
    require_clear as _require_place_contact_clear,
)
from .scene_checked_place import plan_scene_checked_place
from .raw_contacts import raw_contacts_callback as _raw_contacts_callback
from .planning_cpu_diagnostics import (
    initialize as _initialize_planning_cpu,
    callback as _planning_cpu_callback,
    command_target as _planning_cpu_command_target,
    add_status_fields as _add_planning_cpu_fields,
    mark_executor_thread as _mark_planning_cpu_executor,
)
from . import empty_head_timing as _empty_head_timing
from .settled_torso import choose_measured_place_height, require_planned_place_height
from .empty_torso_planning_overlap import (checked_enabled as checked_empty_torso_overlap,
    planning_scope as empty_torso_planning_scope)
from .completed_torso_hold import (TorsoHoldCancellation, TorsoHoldState,
    checked_torso_hold_enabled, checked_torso_hold_retry_enabled, track_completed_torso_hold, require_follow_token_locked,
    try_completed_torso_hold)
from .kinematics import (
    CollisionMesh,
    PreparedTriangleMesh,
    URDFChain,
    interpolate_joint_waypoints,
    load_urdf_collision_meshes,
    oriented_box_intersects_triangles,
    pose_matrix,
    triangle_meshes_intersect,
)
from .motion_profiles import (
    ARM_JOINTS,
    CARRY,
    HOME,
    IK_JOINTS,
    OFFER,
    PREGRASP,
    RIGHT_ARM_JOINTS,
    RIGHT_HOME,
    SUPPORTED_CARRY,
    bin_place_orientations,
    shelf_grasp_orientations,
    shelf_pinch_orientations,
    supported_bin_place_orientations,
)
from .open_gripper_approach import check_open_gripper_approach
from .release_evidence import AttemptIdentity
from .raised_place_finish import (
    checked_enabled as checked_raised_place_finish_enabled, normal_finish as raised_place_normal_finish,
)
from .release_sensor_adapter import (
    observe_measured_open_pose, record_raw_joints, record_raw_odom,
)
from .stock_gripper_close import enabled as _stock_diagnostic, StockGripperClose, contact_evidence as _stock_contact_evidence, measured_position_in_range as _stock_measured_position_in_range
from .lift_pressure_gate import (
    LiftPressureGate, LiftPressureRejected, held_contact_identity_fault,
)
from .runtime_utils import (
    book_model_name,
    is_intentional_gripper_book_contact,
    is_target_book_non_gripper_robot_contact,
    joint_state_cache_key,
    measured_joint_positions,
    normalize_contact_pair,
)
from .target_tracking import (
    TargetBookObservation,
    TargetTrackingResult,
    guard_target_motion,
)

# Import registers PointStamped conversions with tf2_ros in ROS 2 Humble.
import tf2_geometry_msgs  # noqa: F401, E402


class RetainedMotionNotStopped(RuntimeError):
    """A carried action has no confirmed terminal state; recovery must not move."""


LEFT_COLLISION_LINKS = tuple(
    f'arm_left_{index}_link' for index in range(1, 8)
)
RIGHT_COLLISION_LINKS = tuple(
    f'arm_right_{index}_link' for index in range(1, 8)
)
HEAD_COLLISION_LINKS = (
    'head_1_link',
    'head_2_link',
    'head_front_camera_link',
)
HEAD_JOINTS = ('head_1_joint', 'head_2_joint')
CARRIED_COLLISION_LINKS = (
    'base_link',
    'torso_base_link',
    'torso_lift_link',
    *LEFT_COLLISION_LINKS,
    *RIGHT_COLLISION_LINKS,
    *HEAD_COLLISION_LINKS,
)
DIRECTLY_CONNECTED_COLLISION_LINKS = {
    frozenset(('base_link', 'torso_base_link')),
    frozenset(('torso_base_link', 'torso_lift_link')),
    frozenset(('torso_lift_link', 'arm_left_1_link')),
    frozenset(('torso_lift_link', 'arm_right_1_link')),
    frozenset(('torso_lift_link', 'head_1_link')),
    frozenset(('head_1_link', 'head_2_link')),
    frozenset(('head_2_link', 'head_front_camera_link')),
    *(
        frozenset((f'arm_left_{index}_link', f'arm_left_{index + 1}_link'))
        for index in range(1, 7)
    ),
    *(
        frozenset((f'arm_right_{index}_link', f'arm_right_{index + 1}_link'))
        for index in range(1, 7)
    ),
}


@dataclass(frozen=True)
class _TargetTrackingStatus:
    """Latest fail-closed availability edge for the volatile RGB-D track."""

    available: bool
    reason: str
    track_id: str = ''
    track_generation: int = 0
    target_colour: str = ''
    row: int = 0
    stamp_ns: int = -1
    sequence: int = 0


def _contact_force_magnitude(
    contact: object,
    *,
    gripper_is_collision1: bool,
) -> Optional[float]:
    """Return the finite total force applied to the gripper in one contact."""
    wrench_field = (
        'body_1_wrench' if gripper_is_collision1 else 'body_2_wrench'
    )
    wrenches = list(getattr(contact, 'wrenches', ()))
    if not wrenches:
        return None
    total = 0.0
    for joint_wrench in wrenches:
        wrench = getattr(joint_wrench, wrench_field, None)
        force = getattr(wrench, 'force', None)
        values = np.asarray(
            [
                getattr(force, 'x', math.nan),
                getattr(force, 'y', math.nan),
                getattr(force, 'z', math.nan),
            ],
            dtype=float,
        )
        if values.shape == (3,):
            # Same float64 dot and sqrt as NumPy's default vector norm.
            # Avoid its generic array/axis dispatch and three-value reduction.
            if not (math.isfinite(values[0]) and math.isfinite(values[1])
                    and math.isfinite(values[2])):
                return None
            total += float(np.sqrt(np.dot(values, values)))
        else:
            # Retain the original handling of nonstandard field shapes.
            if not np.all(np.isfinite(values)):
                return None
            total += float(np.linalg.norm(values))
    return total if math.isfinite(total) else None


def _message_stamp_ns(
    message: object, fallback_ns: int, *, clamp_small_future: bool = True,
) -> int:
    """Use a valid producer stamp so queued DDS data cannot become fresh."""
    stamp = getattr(getattr(message, 'header', None), 'stamp', None)
    try:
        seconds = int(stamp.sec)
        nanoseconds = int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return int(fallback_ns)
    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        return int(fallback_ns)
    combined = seconds * 1_000_000_000 + nanoseconds
    if combined <= 0:
        return int(fallback_ns)
    if (clamp_small_future
            and int(fallback_ns) < combined <= int(fallback_ns) + 100_000_000):
        # ROS callbacks for /clock and sensor data may be dispatched in either
        # order for the same simulation step.  Clamp only that small positive
        # skew; an old queued producer stamp is always preserved.
        return int(fallback_ns)
    return combined


def _integer_metadata(value: object, name: str, *, minimum: int) -> int:
    """Decode integer-valued JSON/float32 metadata without truncation."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} is not numeric')
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        rounded = int(value)
        if rounded < minimum:
            raise ValueError(f'{name} is below {minimum}')
        return rounded
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} is not numeric') from exc
    rounded = int(round(numeric))
    if not math.isfinite(numeric) or abs(numeric - rounded) > 1e-4:
        raise ValueError(f'{name} is not an integer')
    if rounded < minimum:
        raise ValueError(f'{name} is below {minimum}')
    return rounded


def _decode_target_tracking_status(
    raw: str,
    expected_colour: str,
) -> _TargetTrackingStatus:
    """Decode the producer's edge status; every malformed state is blocked."""
    payload = decode_event(raw)
    event = str(payload.get('event', 'invalid'))
    if event != 'target_tracking':
        reason = str(payload.get('reason', event or 'invalid_tracking_status'))
        return _TargetTrackingStatus(False, reason)
    try:
        if str(payload.get('mode', '')) != 'books':
            raise ValueError('perception mode is not books')
        colour = str(payload['target_colour']).strip().lower()
        if colour != str(expected_colour).strip().lower():
            raise ValueError('target colour does not match the mission')
        track_id = str(payload['track_id']).strip()
        if not track_id:
            raise ValueError('track_id is empty')
        generation = _integer_metadata(
            payload['track_generation'],
            'track_generation',
            minimum=1,
        )
        row = _integer_metadata(payload['row'], 'row', minimum=1)
        if row > 4:
            raise ValueError('row is outside the competition shelf')
        stamp_ns = _integer_metadata(payload['stamp_ns'], 'stamp_ns', minimum=1)
        sequence = _integer_metadata(payload['sequence'], 'sequence', minimum=1)
    except (KeyError, TypeError, ValueError) as exc:
        return _TargetTrackingStatus(
            False,
            f'invalid_target_tracking_status:{exc}',
        )
    return _TargetTrackingStatus(
        True,
        '',
        track_id=track_id,
        track_generation=generation,
        target_colour=colour,
        row=row,
        stamp_ns=stamp_ns,
        sequence=sequence,
    )


def _point_cloud_channel(
    cloud: PointCloud,
    name: str,
    *,
    repeated: bool = True,
) -> np.ndarray:
    """Return one finite, unambiguous five-value tracking channel."""
    matches = [channel for channel in cloud.channels if channel.name == name]
    if len(matches) != 1:
        raise ValueError(f'tracking channel {name!r} is missing or duplicated')
    values = np.asarray(matches[0].values, dtype=float)
    if values.shape != (5,) or not np.all(np.isfinite(values)):
        raise ValueError(f'tracking channel {name!r} must have five finite values')
    if repeated and not np.allclose(values, values[0], rtol=0.0, atol=1e-6):
        raise ValueError(f'tracking channel {name!r} is not repeated consistently')
    return values


def _quaternion_rotation_matrix(quaternion) -> np.ndarray:
    """Return a checked 3x3 rotation matrix for a geometry quaternion."""
    values = np.asarray(
        [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
        dtype=float,
    )
    magnitude = float(np.linalg.norm(values))
    if not np.all(np.isfinite(values)) or magnitude <= 1e-9:
        raise ValueError('target-tracking transform quaternion is invalid')
    x, y, z, w = values / magnitude
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
             2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
             2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
             1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        dtype=float,
    )


class ManipulationNode(Node):
    """Execute head, pick, carry, and place commands with bounded recovery."""

    def __init__(self) -> None:
        super().__init__('erc_manipulation')
        self._declare_parameters()
        self.dry_run = bool(self.get_parameter('dry_run').value)
        self.delivery_evidence_enabled = bool(
            self.get_parameter('delivery_evidence_enabled').value)
        from .head_return_overlap import checked_enabled
        self.head_return_overlap_enabled = checked_enabled(self.get_parameter('head_return_overlap_enabled').value)
        self.empty_head_timing_enabled = _empty_head_timing.checked_enabled(
            self.get_parameter('empty_head_timing_enabled').value)
        self.completed_head_hold_recheck_enabled = checked_enabled(
            self.get_parameter('completed_head_hold_recheck_enabled').value)
        self._empty_head_timing_owner = None
        self._empty_head_timing_limits = None
        self._empty_head_pick_started = False
        self._head_return_state = None
        self.target_colour = str(self.get_parameter('book_colour').value).lower()
        self.timeout = float(self.get_parameter('command_timeout_seconds').value)
        self.tf_timeout = float(self.get_parameter('tf_timeout_seconds').value)
        self.perception_max_age = float(
            self.get_parameter('perception_max_age_seconds').value
        )
        self.perception_wait = float(
            self.get_parameter('perception_wait_seconds').value
        )
        # This independent live stream is the only perception authority used
        # by guarded shelf-edge motion.  Keep the consumer cap at 150 ms even
        # if a deployment accidentally supplies a more permissive parameter.
        configured_tracking_age = float(
            self.get_parameter('target_tracking_maximum_age_seconds').value
        )
        self.target_tracking_max_age = min(0.15, configured_tracking_age)
        self.target_tracking_guard_wait = float(
            self.get_parameter('target_tracking_guard_wait_seconds').value
        )
        self.target_tracking_minimum_rate = float(
            self.get_parameter('target_tracking_minimum_rate_hz').value
        )
        self.target_tracking_minimum_depth_coverage = float(
            self.get_parameter('target_tracking_minimum_depth_coverage').value
        )
        self.target_tracking_maximum_plane_residual = float(
            self.get_parameter(
                'target_tracking_maximum_plane_residual_m'
            ).value
        )
        self.target_tracking_maximum_center_uncertainty = float(
            self.get_parameter(
                'target_tracking_maximum_center_uncertainty_m'
            ).value
        )
        self.target_tracking_maximum_extent_uncertainty = float(
            self.get_parameter(
                'target_tracking_maximum_extent_uncertainty_m'
            ).value
        )
        self.target_tracking_maximum_orientation_uncertainty = float(
            self.get_parameter(
                'target_tracking_maximum_orientation_uncertainty_rad'
            ).value
        )
        if not (
            math.isfinite(configured_tracking_age)
            and math.isfinite(self.target_tracking_guard_wait)
            and math.isfinite(self.target_tracking_minimum_rate)
            and math.isfinite(self.target_tracking_minimum_depth_coverage)
            and math.isfinite(self.target_tracking_maximum_plane_residual)
            and math.isfinite(self.target_tracking_maximum_center_uncertainty)
            and math.isfinite(self.target_tracking_maximum_extent_uncertainty)
            and math.isfinite(
                self.target_tracking_maximum_orientation_uncertainty
            )
            and 0.0 < self.target_tracking_max_age <= 0.15
            and 0.0 < self.target_tracking_guard_wait <= 0.50
            and self.target_tracking_minimum_rate >= 10.0
            and 0.0 < self.target_tracking_minimum_depth_coverage <= 1.0
            and self.target_tracking_maximum_plane_residual > 0.0
            and self.target_tracking_maximum_center_uncertainty > 0.0
            and self.target_tracking_maximum_extent_uncertainty > 0.0
            and self.target_tracking_maximum_orientation_uncertainty > 0.0
        ):
            raise ValueError('target-tracking safety limits are invalid')
        self.gripper_open = float(self.get_parameter('gripper_open_position').value)
        self.gripper_closed = float(self.get_parameter('gripper_closed_position').value)
        self.gripper_preclose = float(
            self.get_parameter('gripper_preclose_position').value
        )
        self.gripper_preload = float(
            self.get_parameter('gripper_preload_position').value
        )
        self.gripper_transport_lock = float(
            self.get_parameter('gripper_transport_lock_position').value
        )
        self.grasp_min_position = float(
            self.get_parameter('grasp_min_position').value
        )
        self.grasp_min_margin = float(
            self.get_parameter('grasp_min_margin').value
        )
        self.grasp_max_position = float(
            self.get_parameter('grasp_max_position').value
        )
        self.pregrasp_offset = float(self.get_parameter('pregrasp_offset').value)
        self.grasp_depth_offset = float(self.get_parameter('grasp_depth_offset').value)
        self.top_row_grasp_depth_offset = float(
            self.get_parameter('top_row_grasp_depth_offset').value
        )
        self.shelf_side_cradle_enabled = bool(
            self.get_parameter('shelf_side_cradle_enabled').value
        )
        self.top_row_grasp_vertical_offset = float(
            self.get_parameter('top_row_grasp_vertical_offset').value
        )
        self.top_row_grasp_lateral_offset = float(
            self.get_parameter('top_row_grasp_lateral_offset').value
        )
        self.lift_first_extraction_enabled = bool(
            self.get_parameter('lift_first_extraction_enabled').value
        )
        self.preclose_aperture_geometry_enabled = bool(
            self.get_parameter('preclose_aperture_geometry_enabled').value)
        self._preclose_aperture_generation = 0
        self._preclose_aperture_state = None
        self._preclose_aperture_checked = None
        self._preclose_aperture_dispatched = False
        self._preclose_aperture_model = None
        self.lift_first_extraction_lift_m = float(
            self.get_parameter('lift_first_extraction_lift_m').value
        )
        self.lift_first_minimum_contact_force = float(
            self.get_parameter('lift_first_minimum_contact_force').value
        )
        self.lift_first_maximum_gripper_velocity_mps = float(
            self.get_parameter('lift_first_maximum_gripper_velocity_mps').value)
        if not (math.isfinite(self.lift_first_maximum_gripper_velocity_mps)
                and 0 < self.lift_first_maximum_gripper_velocity_mps <= 1e-5):
            raise ValueError('lift-first gripper velocity must be within (0, 10e-6] m/s')
        self.top_row_loaded_clearance_lift = float(
            self.get_parameter('top_row_loaded_clearance_lift_distance').value
        )
        self.retreat_distance = float(self.get_parameter('retreat_distance').value)
        self.place_clearance = float(self.get_parameter('place_clearance').value)
        self.book_centered_place_enabled = bool(self.get_parameter('book_centered_place_enabled').value)
        self.table_scene_required = bool(self.get_parameter('table_scene_required').value)
        self.bin_scene_required = bool(self.get_parameter('bin_scene_required').value)
        if self.bin_scene_required and not (self.table_scene_required and self.book_centered_place_enabled):
            raise ValueError('bin_scene_required requires table_scene_required and book_centered_place_enabled')
        self.placement_transport_speed_scale = checked_arm_speed_scale(
            self.get_parameter('placement_transport_speed_scale').value)
        self.loaded_place_speed_scale_cap = checked_arm_speed_scale(
            self.get_parameter('loaded_place_speed_scale_cap').value)
        self.bin_clearance_timing_enabled = checked_bin_clearance_timing_enabled(
            self.get_parameter('bin_clearance_timing_enabled').value)
        self.place_finish_at_release_enabled = release_pose_finish.checked_enabled(
            self.get_parameter('place_finish_at_release_enabled').value)
        self.release_only_place_planning_enabled = release_only_place_planning.checked_enabled(
            self.get_parameter('release_only_place_planning_enabled').value)
        self._release_pose_owner = None
        self._release_pose_fault_latched = None
        self.withdrawal_half_timing_enabled = checked_withdrawal_half_timing_enabled(
            self.get_parameter('withdrawal_half_timing_enabled').value)
        self.initial_stow_serial_timing_enabled = _initial_stow_serial.checked_enabled(
            self.get_parameter('initial_stow_serial_timing_enabled').value)
        self._initial_stow_serial_owner = None
        self._initial_stow_serial_command_seen = False
        self._initial_stow_serial_limits = None
        self.withdrawal_quarter_timing_enabled = self.get_parameter(
            'withdrawal_quarter_timing_enabled').value
        if type(self.withdrawal_quarter_timing_enabled) is not bool:
            raise ValueError('withdrawal_quarter_timing_enabled must be Boolean')
        if self.withdrawal_quarter_timing_enabled and not self.withdrawal_half_timing_enabled:
            raise ValueError('withdrawal quarter timing requires withdrawal_half_timing_enabled')
        self.withdrawal_speed_scale = checked_withdrawal_speed_scale(
            self.get_parameter('withdrawal_speed_scale').value)
        self.additional_arm_time_scale = checked_additional_arm_time_scale(
            self.get_parameter('additional_arm_time_scale').value)
        self.compact_extension_half_timing_enabled = self.get_parameter(
            'compact_extension_half_timing_enabled').value
        if type(self.compact_extension_half_timing_enabled) is not bool:
            raise ValueError('compact_extension_half_timing_enabled must be Boolean')
        if self.compact_extension_half_timing_enabled and self.additional_arm_time_scale != 2.0:
            raise ValueError('compact extension half timing requires additional_arm_time_scale=2')
        self.empty_pickup_setup_retiming_enabled = checked_empty_pickup_setup_retiming_enabled(
            self.get_parameter('empty_pickup_setup_retiming_enabled').value)
        self.geometry_process_workers = checked_geometry_process_workers(
            self.get_parameter('geometry_process_workers').value)
        self.place_parallel_geometry_enabled = checked_place_parallel_geometry_enabled(
            self.get_parameter('place_parallel_geometry_enabled').value)
        self.place_carried_volume_parallel_enabled = checked_place_carried_volume_parallel_enabled(
            self.get_parameter('place_carried_volume_parallel_enabled').value)
        if self.place_carried_volume_parallel_enabled and not self.place_parallel_geometry_enabled:
            raise ValueError('parallel carried volume requires parallel PLACE geometry')
        self.pickup_parallel_geometry_enabled = checked_pickup_parallel_geometry_enabled(
            self.get_parameter('pickup_parallel_geometry_enabled').value)
        self.empty_pickup_parallel_geometry_enabled = checked_empty_pickup_parallel_geometry_enabled(
            self.get_parameter('empty_pickup_parallel_geometry_enabled').value)
        if self.empty_pickup_parallel_geometry_enabled and not self.pickup_parallel_geometry_enabled:
            raise ValueError('empty parallel geometry requires pickup parallel geometry')
        self.pickup_post_retreat_parallel_geometry_enabled = checked_pickup_post_retreat_parallel_geometry_enabled(
            self.get_parameter('pickup_post_retreat_parallel_geometry_enabled').value)
        if self.pickup_post_retreat_parallel_geometry_enabled and not self.pickup_parallel_geometry_enabled:
            raise ValueError('post-retreat parallel geometry requires pickup parallel geometry')
        self.supported_compact_minimum_segment_seconds = checked_supported_compact_minimum_seconds(
            self.get_parameter('supported_compact_minimum_segment_seconds').value)
        self.scene_cartesian_wall_seconds = float(
            self.get_parameter('scene_cartesian_wall_seconds').value)
        if (not math.isfinite(self.scene_cartesian_wall_seconds)
                or not 0.0 < self.scene_cartesian_wall_seconds <= 1800.0):
            raise ValueError('scene_cartesian_wall_seconds must be finite and within (0, 1800]')
        self.book_centered_place_height_above_point = float(self.get_parameter('book_centered_place_height_above_point_m').value)
        self.book_centered_place_approach_x = float(self.get_parameter('book_centered_place_approach_x_m').value)
        self.book_centered_place_clearance_height = float(self.get_parameter('book_centered_place_clearance_height_m').value)
        if not all(np.isfinite(value) and value > 0.0 for value in (
            self.book_centered_place_height_above_point, self.book_centered_place_approach_x,
            self.book_centered_place_clearance_height,
        )):
            raise ValueError('book-centred placement height and approach distance must be finite and positive')
        self.position_tolerance = float(
            self.get_parameter('ik_position_tolerance').value
        )
        self.orientation_tolerance = float(
            self.get_parameter('ik_orientation_tolerance').value
        )
        self.pick_position_tolerance = float(
            self.get_parameter('pick_ik_position_tolerance').value
        )
        self.pick_orientation_tolerance = float(
            self.get_parameter('pick_ik_orientation_tolerance').value
        )
        if not 0.0 < self.pick_position_tolerance <= 0.0005:
            raise ValueError('pick IK position tolerance must be within (0, 0.0005]')
        if not 0.0 < self.pick_orientation_tolerance <= 0.01:
            raise ValueError('pick IK orientation tolerance must be within (0, 0.01]')
        self.place_joint_limit_margin = float(
            self.get_parameter('place_joint_limit_margin_radians').value
        )
        self.cartesian_clearance = float(
            self.get_parameter('cartesian_clearance_distance').value
        )
        self.top_row_cartesian_clearance = float(
            self.get_parameter('top_row_cartesian_clearance_distance').value
        )
        self.cartesian_step = float(self.get_parameter('cartesian_step').value)
        self.cartesian_joint_step = float(
            self.get_parameter('cartesian_joint_step_limit').value
        )
        self.lower_shelf_pick_enabled = bool(
            self.get_parameter('lower_shelf_pick_enabled').value)
        self.pick_torso_height = float(
            self.get_parameter('pick_torso_height').value
        )
        self.place_torso_height = float(
            self.get_parameter('place_torso_height').value
        )
        self.empty_torso_planning_overlap_enabled = checked_empty_torso_overlap(
            self.get_parameter('empty_torso_planning_overlap_enabled').value)
        self._empty_torso_planning_owner = None
        self.settled_place_torso_skip_enabled = checked_torso_hold_enabled(
            self.get_parameter('settled_place_torso_skip_enabled').value)
        self.settled_place_torso_retry_enabled = checked_torso_hold_retry_enabled(
            self.get_parameter('settled_place_torso_retry_enabled').value)
        if self.empty_torso_planning_overlap_enabled and not self.settled_place_torso_skip_enabled:
            raise ValueError('empty torso overlap requires settled torso tracking')
        if self.settled_place_torso_retry_enabled and not self.settled_place_torso_skip_enabled:
            raise ValueError('settled torso retry requires settled torso skip')
        self.measured_place_height_enabled = bool(
            self.get_parameter('measured_place_height_enabled').value)
        self.preopen_stationary_enabled = bool(
            self.get_parameter('preopen_stationary_enabled').value)
        self.place_transition_stop_enabled = _place_transition_stop.checked_enabled(
            self.get_parameter('place_transition_stop_enabled').value)
        self._place_transition_stop_owner = None
        self._place_transition_scene_failure = None
        self.release_only_clearance_timing_enabled = _release_clearance.checked_enabled(
            self.get_parameter('release_only_clearance_timing_enabled').value)
        self._release_only_clearance_owner = None
        if self.release_only_clearance_timing_enabled and (
                not self.release_only_place_planning_enabled or not self.place_transition_stop_enabled
                or self.bin_clearance_timing_enabled or self.loaded_place_speed_scale_cap != 2.0
                or self.additional_arm_time_scale != 2.0):
            raise ValueError('release-only clearance requires exclusive release-only cap2 transition mode')
        if self.place_transition_stop_enabled and self.loaded_place_speed_scale_cap != 2.0:
            raise ValueError('place transition stop requires loaded_place_speed_scale_cap=2')
        self.place_finish_keep_torso_height_enabled = checked_raised_place_finish_enabled(
            self.get_parameter('place_finish_keep_torso_height_enabled').value)
        self.gripper_settle = float(
            self.get_parameter('gripper_settle_seconds').value
        )
        self.grasp_contact_max_age = float(
            self.get_parameter('grasp_contact_max_age_seconds').value
        )
        self.adaptive_close_step = float(
            self.get_parameter('adaptive_gripper_close_step').value
        )
        self.adaptive_step_motion = float(
            self.get_parameter(
                'adaptive_gripper_step_motion_seconds'
            ).value
        )
        self.adaptive_step_settle = float(
            self.get_parameter(
                'adaptive_gripper_step_settle_seconds'
            ).value
        )
        self.adaptive_preload_distance = float(
            self.get_parameter(
                'adaptive_gripper_preload_distance'
            ).value
        )
        self.adaptive_confirmation_seconds = float(
            self.get_parameter(
                'adaptive_gripper_confirmation_seconds'
            ).value
        )
        self.adaptive_contact_force_minimum = float(
            self.get_parameter(
                'adaptive_gripper_contact_force_minimum'
            ).value
        )
        self.adaptive_contact_force_maximum = float(
            self.get_parameter(
                'adaptive_gripper_contact_force_maximum'
            ).value
        )
        self.adaptive_contact_samples = int(
            self.get_parameter('adaptive_gripper_contact_samples').value
        )
        self.adaptive_contact_max_age = float(
            self.get_parameter(
                'adaptive_gripper_contact_max_age_seconds'
            ).value
        )
        self.adaptive_contact_max_gap = float(
            self.get_parameter(
                'adaptive_gripper_contact_max_gap_seconds'
            ).value
        )
        self.adaptive_contact_min_span = float(
            self.get_parameter(
                'adaptive_gripper_contact_min_span_seconds'
            ).value
        )
        self.adaptive_contact_max_skew = float(
            self.get_parameter(
                'adaptive_gripper_contact_max_skew_seconds'
            ).value
        )
        self.adaptive_velocity_tolerance = float(
            self.get_parameter(
                'adaptive_gripper_velocity_tolerance'
            ).value
        )
        self.adaptive_effort_maximum = float(
            self.get_parameter(
                'adaptive_gripper_effort_maximum'
            ).value
        )
        self.adaptive_effort_delta_maximum = float(
            self.get_parameter(
                'adaptive_gripper_effort_delta_maximum'
            ).value
        )
        self.adaptive_unilateral_travel_limit = float(
            self.get_parameter(
                'adaptive_gripper_unilateral_travel_limit'
            ).value
        )
        self.adaptive_endpoint_tolerance = float(
            self.get_parameter(
                'adaptive_gripper_endpoint_tolerance'
            ).value
        )
        self.adaptive_start_tolerance = float(
            self.get_parameter(
                'adaptive_gripper_start_tolerance'
            ).value
        )
        self.raw_contacts_enabled = bool(
            self.get_parameter('raw_contacts_enabled').value)
        self.stock_gripper_close_diagnostic_enabled = bool(
            self.get_parameter('stock_gripper_close_diagnostic_enabled').value)
        self.stock_gripper_close_target_position = float(
            self.get_parameter('stock_gripper_close_target_position').value)
        if (not math.isfinite(self.stock_gripper_close_target_position)
                or not 0.0 <= self.stock_gripper_close_target_position <= .069):
            raise ValueError('stock close target must be finite and within public range [0, .069]')
        self._stock_close_controller = None
        self.fine_gripper_close_enabled = bool(
            self.get_parameter('fine_gripper_close_enabled').value)
        if self.stock_gripper_close_diagnostic_enabled and (
                self.fine_gripper_close_enabled or not self.lift_first_extraction_enabled
                or self.preclose_aperture_geometry_enabled):
            raise ValueError('stock diagnostic requires lift-first, fine off and ordinary measured geometry')
        from .fine_gripper_close import FineGripLimits
        self.fine_gripper_limits = FineGripLimits(
            maximum_force=self.adaptive_contact_force_maximum,
            minimum_force=float(self.get_parameter('fine_gripper_minimum_force').value),
            step_wall_seconds=float(self.get_parameter('fine_gripper_step_wall_seconds').value),
            total_wall_seconds=float(self.get_parameter('fine_gripper_wall_timeout_seconds').value),
            micro_position_error=float(self.get_parameter('fine_gripper_micro_position_error_m').value),
            micro_stationary_velocity=float(self.get_parameter('fine_gripper_micro_stationary_velocity_mps').value),
            micro_preload_steps=self.get_parameter('fine_gripper_micro_preload_steps').value,
            micro_preload_step=float(self.get_parameter('fine_gripper_micro_preload_step_m').value),
            micro_motion_seconds=float(self.get_parameter('fine_gripper_micro_motion_seconds').value),
            micro_preload_limit=float(self.get_parameter('fine_gripper_micro_preload_limit_m').value))
        if self.preclose_aperture_geometry_enabled and not (
                self.lift_first_extraction_enabled and self.fine_gripper_close_enabled):
            raise ValueError('preclose aperture geometry requires lift-first and guarded fine closure')
        self._fine_gripper_controller = None
        self.carried_book_dimensions = np.asarray(
            self.get_parameter('carried_book_dimensions').value,
            dtype=float,
        )
        self.carried_book_padding = float(
            self.get_parameter('carried_book_padding').value
        )
        self.carried_shelf_margin = float(
            self.get_parameter('carried_shelf_margin').value
        )
        self.carried_shelf_retreat_clearance = float(
            self.get_parameter('carried_shelf_retreat_clearance_distance').value
        )
        self.carried_cradle_extension = float(
            self.get_parameter('carried_cradle_extension_distance').value
        )
        self.carried_cradle_retraction = float(
            self.get_parameter('carried_cradle_retraction_distance').value
        )
        self.carried_cradle_roll = float(
            self.get_parameter('carried_cradle_roll_radians').value
        )
        self.carried_cradle_transfer = float(
            self.get_parameter('carried_cradle_transfer_seconds').value
        )
        self.carried_supported_jaw_vertical_component = float(
            self.get_parameter(
                'carried_supported_jaw_vertical_component'
            ).value
        )
        self.carried_transition_samples = int(
            self.get_parameter('carried_transition_samples').value
        )
        self.carried_orientation_step_limit = float(
            self.get_parameter('carried_orientation_step_limit').value
        )
        self.carried_navigation_radius_limit = float(
            self.get_parameter('carried_navigation_radius_limit').value
        )
        self.carried_maximum_tilt = float(
            self.get_parameter('carried_maximum_tilt_radians').value
        )
        if (
            self.carried_book_dimensions.shape != (3,)
            or not np.all(np.isfinite(self.carried_book_dimensions))
            or np.any(self.carried_book_dimensions <= 0.0)
        ):
            raise ValueError('carried_book_dimensions must contain three positive values')
        if (
            not np.isfinite(self.carried_book_padding)
            or not np.isfinite(self.carried_shelf_margin)
            or not np.isfinite(self.carried_shelf_retreat_clearance)
            or not np.isfinite(self.carried_cradle_extension)
            or not np.isfinite(self.carried_cradle_retraction)
            or not np.isfinite(self.carried_cradle_roll)
            or not np.isfinite(self.carried_cradle_transfer)
            or not np.isfinite(self.carried_supported_jaw_vertical_component)
            or self.carried_book_padding < 0.0
            or self.carried_shelf_margin < 0.0
            or self.carried_shelf_retreat_clearance <= 0.0
            or not 0.10 <= self.carried_cradle_extension <= 0.20
            or self.carried_cradle_retraction <= 0.0
            or not -np.pi < self.carried_cradle_roll < 0.0
            or not 0.5 <= self.carried_cradle_transfer <= 5.0
            or not 0.0 < self.carried_supported_jaw_vertical_component <= 1.0
        ):
            raise ValueError(
                'carried-book margins must be finite and nonnegative, and '
                'retreat/cradle geometry and transfer time must be finite and valid'
            )
        if self.carried_transition_samples < 3:
            raise ValueError('carried_transition_samples must be at least three')
        if (
            not np.isfinite(self.place_joint_limit_margin)
            or self.place_joint_limit_margin < 0.0
        ):
            raise ValueError(
                'place_joint_limit_margin_radians must be finite and nonnegative'
            )
        if (
            not np.isfinite(self.carried_orientation_step_limit)
            or self.carried_orientation_step_limit <= 0.0
        ):
            raise ValueError('carried_orientation_step_limit must be positive')
        if (
            not np.isfinite(self.carried_navigation_radius_limit)
            or self.carried_navigation_radius_limit <= 0.0
        ):
            raise ValueError('carried_navigation_radius_limit must be positive')
        if not 0.0 < self.carried_maximum_tilt < 0.5 * np.pi:
            raise ValueError('carried_maximum_tilt_radians must be between 0 and pi/2')
        if (
            not np.isfinite(self.gripper_settle)
            or not np.isfinite(self.grasp_contact_max_age)
            or self.gripper_settle <= 0.0
            or self.grasp_contact_max_age <= 0.0
        ):
            raise ValueError('gripper settle and contact-age times must be positive')
        adaptive_scalars = (
            self.adaptive_close_step,
            self.adaptive_step_motion,
            self.adaptive_step_settle,
            self.adaptive_preload_distance,
            self.adaptive_confirmation_seconds,
            self.adaptive_contact_force_minimum,
            self.adaptive_contact_force_maximum,
            self.adaptive_contact_max_age,
            self.adaptive_contact_max_gap,
            self.adaptive_contact_min_span,
            self.adaptive_contact_max_skew,
            self.adaptive_velocity_tolerance,
            self.adaptive_effort_maximum,
            self.adaptive_effort_delta_maximum,
            self.adaptive_unilateral_travel_limit,
            self.adaptive_endpoint_tolerance,
            self.adaptive_start_tolerance,
        )
        if not all(math.isfinite(value) for value in adaptive_scalars):
            raise ValueError('adaptive gripper limits must be finite')
        if not (
            0.0 < self.adaptive_close_step <= 0.003
            and 0.0 < self.adaptive_step_motion <= 1.0
            and 0.0 <= self.adaptive_step_settle <= 1.0
            and 0.0 < self.adaptive_preload_distance <= 0.003
            and 0.05 <= self.adaptive_confirmation_seconds <= 0.50
            and 0.0 < self.adaptive_contact_force_minimum
            < self.adaptive_contact_force_maximum
            and self.adaptive_contact_samples >= 3
            and 0.0 < self.adaptive_contact_max_age <= 0.20
            and 0.0 < self.adaptive_contact_max_gap
            <= self.adaptive_contact_max_age
            and 0.0 < self.adaptive_contact_min_span
            <= self.adaptive_contact_max_age
            and 0.0 < self.adaptive_contact_max_skew
            <= self.adaptive_contact_max_age
            and self.adaptive_confirmation_seconds
            >= self.adaptive_contact_min_span
            and self.adaptive_velocity_tolerance > 0.0
            and self.adaptive_effort_maximum > 0.0
            and self.adaptive_effort_delta_maximum > 0.0
            and self.adaptive_unilateral_travel_limit
            >= self.adaptive_close_step
            and 0.0 < self.adaptive_endpoint_tolerance
            <= self.adaptive_close_step
            and 0.0 < self.adaptive_start_tolerance <= 0.005
        ):
            raise ValueError('adaptive gripper safety limits are invalid')
        if self.fine_gripper_close_enabled and not (
                self.adaptive_contact_force_maximum == self.fine_gripper_limits.maximum_force
                and self.adaptive_contact_force_minimum <= self.fine_gripper_limits.minimum_force
                and self.grasp_min_position + self.grasp_min_margin <= self.fine_gripper_limits.floor
                and self.fine_gripper_limits.floor < self.grasp_max_position):
            raise ValueError('fine gripper profile conflicts with force or width guards')
        if not (math.isfinite(self.lift_first_minimum_contact_force)
                and 2.5 <= self.lift_first_minimum_contact_force
                and (not self.lift_first_extraction_enabled
                     or self.lift_first_minimum_contact_force < self.adaptive_contact_force_maximum)):
            raise ValueError('lift-first pressure minimum must be >=2.5 N and below the force ceiling')

        if not (
            0.0 <= self.gripper_closed
            < self.grasp_min_position
            < self.grasp_max_position
            < self.gripper_open
        ):
            raise ValueError('grasp position band must lie between closed and open')
        if not (
            self.gripper_closed
            <= self.gripper_transport_lock
            < self.grasp_min_position + self.grasp_min_margin
            < self.gripper_preload
            < self.gripper_preclose
            < self.grasp_max_position
        ):
            raise ValueError(
                'gripper transport lock must lie below the measured grasp '
                'band and preload/preclose positions must lie inside it'
            )
        if not (
            np.isfinite(self.cartesian_clearance)
            and np.isfinite(self.top_row_cartesian_clearance)
            and 0.0 < self.cartesian_clearance <= self.top_row_cartesian_clearance
        ):
            raise ValueError(
                'top-row Cartesian clearance must be finite, positive, and no '
                'smaller than the normal clearance'
            )
        if (
            not np.isfinite(self.top_row_grasp_vertical_offset)
            or abs(self.top_row_grasp_vertical_offset) > 0.05
        ):
            raise ValueError(
                'top_row_grasp_vertical_offset must be finite and within 0.05 m'
            )
        if (
            not np.isfinite(self.top_row_grasp_lateral_offset)
            or abs(self.top_row_grasp_lateral_offset) > 0.01
        ):
            raise ValueError(
                'top_row_grasp_lateral_offset must be finite and within 0.01 m'
            )
        if (
            not np.isfinite(self.top_row_loaded_clearance_lift)
            or not 0.0 <= self.top_row_loaded_clearance_lift <= 0.03
        ):
            raise ValueError(
                'top_row_loaded_clearance_lift_distance must be finite and '
                'within 0.03 m'
            )
        if (not np.isfinite(self.lift_first_extraction_lift_m)
                or not 0.0 < self.lift_first_extraction_lift_m <= .020):
            raise ValueError('lift_first_extraction_lift_m must be positive and at most 20 mm')
        if not (
            0.0 <= self.grasp_min_margin
            < self.gripper_preclose - self.grasp_min_position
        ):
            raise ValueError('grasp_min_margin must fit below gripper_preclose_position')
        if self.grasp_depth_offset <= 0.0 or self.top_row_grasp_depth_offset <= 0.0:
            raise ValueError('grasp depth offsets must be positive')
        self.book_overview_tilt = float(
            self.get_parameter('book_overview_tilt').value
        )
        self.book_row_tilts = [
            float(value) for value in self.get_parameter('book_row_tilts').value
        ]
        if len(self.book_row_tilts) != 4:
            raise ValueError('book_row_tilts must contain four top-to-bottom tilts')

        urdf = (
            Path(get_package_share_directory('erc_description'))
            / 'urdf'
            / 'tiago_pro.urdf'
        )
        if (self.placement_transport_speed_scale > 1.0 or self.withdrawal_speed_scale > 1.0
                or self.empty_pickup_setup_retiming_enabled or getattr(self, 'bin_clearance_timing_enabled', False)
                or getattr(self, 'withdrawal_half_timing_enabled', False)
                or getattr(self, 'compact_extension_half_timing_enabled', False)
                or getattr(self, 'release_only_clearance_timing_enabled', False)):
            from .empty_pickup_collision import joint_velocity_limits
            # Same deployed URDF as the model below; optional-only and once.
            self._faster_arm_velocity_limits = tuple(float(v) for v in joint_velocity_limits(urdf)[1:])
            self._faster_arm_velocity_urdf = str(urdf)
        if self.empty_head_timing_enabled:
            self._empty_head_timing_limits = _empty_head_timing.load_limits(urdf)
        self.chain = URDFChain.from_urdf(
            urdf,
            'base_footprint',
            'gripper_left_grasping_link',
            IK_JOINTS,
        )
        self.right_chain = URDFChain.from_urdf(
            urdf,
            'base_footprint',
            'gripper_right_grasping_link',
            ('torso_lift_joint', *RIGHT_ARM_JOINTS),
        )
        self.head_chain = URDFChain.from_urdf(
            urdf,
            'base_footprint',
            'head_front_camera_link',
            ('torso_lift_joint', *HEAD_JOINTS),
        )
        self.carried_collision_meshes: Tuple[CollisionMesh, ...] = (
            load_urdf_collision_meshes(
                urdf,
                CARRIED_COLLISION_LINKS,
                get_package_share_directory,
                immutable_local=True,
            )
        )
        initialize_place_geometry_backend(self, get_package_share_directory)
        initialize_pickup_geometry_backend(self, get_package_share_directory)
        if self.initial_stow_serial_timing_enabled:
            self._initial_stow_serial_limits = _initial_stow_serial.load_limits(urdf)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.joints: Dict[str, float] = {}
        self.latest_book: Optional[PointStamped] = None
        self.latest_bin: Optional[PointStamped] = None
        self._latest_target_tracking: Optional[PointCloud] = None
        self._target_tracking_status = _TargetTrackingStatus(
            False,
            'target_tracking_status_not_received',
        )
        self._busy = False
        self._cancel = (TorsoHoldCancellation() if self.settled_place_torso_skip_enabled
                        else threading.Event())
        self._torso_hold_state = TorsoHoldState()
        self._goal_handles = []
        self._pending_retained_acceptances = set()
        self._lock = threading.Lock()
        # Adaptive publications and emergency holds use one ordering. Sensor
        # callbacks release _lock before acquiring this command lock.
        self._adaptive_command_lock = threading.RLock()
        self._adaptive_motion_halt_reason: Optional[str] = None
        self._adaptive_hold_sent = False
        self._adaptive_motion_started = False
        self._right_parked = False
        self._contact_generation = 0
        self._contact_epoch = 0
        self._target_book_model: Optional[str] = None
        self._book_contact_samples: Dict[str, List[int]] = {}
        self._book_contact_force_samples: Dict[
            str, List[deque[ForceSample]]
        ] = {}
        self._gripper_feedback_samples: deque[GripperFeedback] = deque(
            maxlen=128
        )
        self._clear_target_contact_samples()
        self._target_robot_contact_latched = False
        self._payload_hazard_latched: Optional[str] = None
        self._payload_monitor_enabled = False
        self._retention_probe_active = False
        self._payload_robot_watchdog_enabled = False
        self._held_book_corners: Optional[np.ndarray] = None
        self._carried_staging_solution: Optional[np.ndarray] = None
        self._cached_post_retreat_plan: Optional[Dict[str, object]] = None
        self._post_retreat_shelf_front_x: Optional[float] = None
        self._gravity_supported_payload = False
        self._supported_post_retreat_staging_required = False
        self._transport_lock_engaged = False
        self._gripper_open_confirmed = False
        self._adaptive_close_active = False
        self._adaptive_overload_latched: Optional[str] = None
        self._adaptive_effort_baseline_value = math.nan
        self._self_collision_cache: Dict[
            bytes, Optional[Tuple[str, str]]
        ] = {}
        self._static_self_collision_cache: Dict[
            bytes, Optional[Tuple[str, str]]
        ] = {}

        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.get_parameter('left_arm_action').value),
        )
        self.right_arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.get_parameter('right_arm_action').value),
        )
        self.head_client = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.get_parameter('head_action').value),
        )
        self.torso_client = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.get_parameter('torso_action').value),
        )
        self.gripper_pub = self.create_publisher(
            JointTrajectory,
            str(self.get_parameter('left_gripper_topic').value),
            10,
        )
        try:
            _initialize_planning_cpu(self)
        except Exception:
            pass
        self.status_pub = self.create_publisher(
            String, '/erc/manipulation/status', TRANSIENT_RELIABLE_QOS
        )
        self.create_subscription(JointState, '/joint_states', _planning_cpu_callback(self, '_on_joint_state', self._on_joint_state), SENSOR_QOS)
        self.create_subscription(Odometry, '/odom', _planning_cpu_callback(self, '_on_staging_odom', self._on_staging_odom), SENSOR_QOS)
        if getattr(self, 'raw_contacts_enabled', False):
            self.create_subscription(
                Contacts, '/contacts',
                _planning_cpu_callback(self, '_on_contacts',
                    _raw_contacts_callback(self, self._on_contacts, '/contacts')),
                SENSOR_QOS, raw=True,
            )
        else:
            self.create_subscription(Contacts, '/contacts', _planning_cpu_callback(self, '_on_contacts', self._on_contacts), SENSOR_QOS)
        if self.table_scene_required:
            # Bin-side sensing also covers fixed tool collisions that have no
            # individual robot contact sensor. Do not duplicate grip samples.
            if getattr(self, 'raw_contacts_enabled', False):
                self.create_subscription(
                    Contacts, '/bin_contacts',
                    _planning_cpu_callback(self, 'bin_contacts',
                        _raw_contacts_callback(self,
                            lambda message: _observe_place_contacts(self, message, '/bin_contacts'),
                            '/bin_contacts')),
                    SENSOR_QOS, raw=True,
                )
            else:
                self.create_subscription(
                    Contacts, '/bin_contacts',
                    _planning_cpu_callback(self, 'bin_contacts',
                        lambda message: _observe_place_contacts(self, message, '/bin_contacts')),
                    SENSOR_QOS,
                )
        self.create_subscription(
            PointStamped,
            '/erc/perception/target_book',
            _planning_cpu_callback(self, '_on_book', self._on_book),
            TRANSIENT_RELIABLE_QOS,
        )
        # Both topics are deliberately volatile.  Rejoining manipulation must
        # observe a live track and its availability edge, never a retained pose.
        self.create_subscription(
            PointCloud,
            '/erc/perception/target_book_observation',
            _planning_cpu_callback(self, '_on_target_tracking', self._on_target_tracking),
            RELIABLE_QOS,
        )
        self.create_subscription(
            String,
            '/erc/perception/target_book_tracking_status',
            _planning_cpu_callback(self, '_on_target_tracking_status', self._on_target_tracking_status),
            RELIABLE_QOS,
        )
        self.create_subscription(
            PointStamped,
            '/erc/perception/bin',
            _planning_cpu_callback(self, '_on_bin', self._on_bin),
            TRANSIENT_RELIABLE_QOS,
        )
        self.create_subscription(
            String, '/erc/perception/status', _planning_cpu_callback(self, '_on_bin_status', self._on_bin_status),
            TRANSIENT_RELIABLE_QOS,
        )
        self.create_subscription(
            String,
            '/erc/manipulation/command',
            _planning_cpu_callback(self, '_on_command', self._on_command),
            RELIABLE_QOS,
        )
        self.create_timer(1.0, _planning_cpu_callback(self, '_announce_ready', self._announce_ready))
        self.create_timer(0.05, _planning_cpu_callback(self, '_monitor_held_payload', self._monitor_held_payload))
        self._ready_announced = False

    def _declare_parameters(self) -> None:
        values = {
            'dry_run': False,
            'delivery_evidence_enabled': False,
            'head_return_overlap_enabled': False,
            'empty_head_timing_enabled': False,
            'completed_head_hold_recheck_enabled': False,
            'book_colour': 'red',
            'command_timeout_seconds': 120.0,
            'tf_timeout_seconds': 1.0,
            'perception_max_age_seconds': 1.5,
            'perception_wait_seconds': 2.0,
            'target_tracking_maximum_age_seconds': 0.15,
            'target_tracking_guard_wait_seconds': 0.25,
            'target_tracking_minimum_rate_hz': 10.0,
            'target_tracking_minimum_depth_coverage': 0.65,
            'target_tracking_maximum_plane_residual_m': 0.002,
            'target_tracking_maximum_center_uncertainty_m': 0.0008,
            'target_tracking_maximum_extent_uncertainty_m': 0.0016,
            'target_tracking_maximum_orientation_uncertainty_rad': 0.05,
            'left_arm_action': '/arm_left_controller/follow_joint_trajectory',
            'right_arm_action': '/arm_right_controller/follow_joint_trajectory',
            'left_gripper_topic': '/gripper_left_controller/joint_trajectory',
            'head_action': '/head_controller/follow_joint_trajectory',
            'torso_action': '/torso_controller/follow_joint_trajectory',
            'book_overview_tilt': -0.10,
            'book_row_tilts': [0.20, 0.00, -0.40, -0.70],
            'gripper_open_position': 0.069,
            'gripper_closed_position': 0.0,
            'gripper_preclose_position': 0.025,
            'gripper_preload_position': 0.018,
            'gripper_transport_lock_position': 0.015,
            'grasp_min_position': 0.0155,
            'grasp_min_margin': 0.0005,
            'grasp_max_position': 0.05,
            'pregrasp_offset': 0.14,
            'grasp_depth_offset': 0.025,
            'top_row_grasp_depth_offset': 0.025,
            'shelf_side_cradle_enabled': False,
            'lift_first_extraction_enabled': False,
            'preclose_aperture_geometry_enabled': False,
            'lift_first_minimum_contact_force': 3.5,
            'lift_first_maximum_gripper_velocity_mps': 0.000002,
            'lift_first_extraction_lift_m': .005,
            'top_row_grasp_vertical_offset': -0.015,
            'top_row_grasp_lateral_offset': -0.001,
            'top_row_loaded_clearance_lift_distance': 0.0,
            'retreat_distance': 0.18,
            'place_clearance': 0.22,
            'book_centered_place_enabled': False,
            'table_scene_required': False,
            'bin_scene_required': False,
            'placement_transport_speed_scale': 1.0,
            'loaded_place_speed_scale_cap': 3.0,
            'bin_clearance_timing_enabled': False,
            'place_finish_at_release_enabled': False,
            'release_only_place_planning_enabled': False,
            'withdrawal_half_timing_enabled': False,
            'withdrawal_quarter_timing_enabled': False,
            'initial_stow_serial_timing_enabled': False,
            'withdrawal_speed_scale': 1.0,
            'additional_arm_time_scale': 1.0,
            'compact_extension_half_timing_enabled': False,
            'empty_pickup_setup_retiming_enabled': False,
            'geometry_process_workers': 4,
            'place_parallel_geometry_enabled': False,
            'place_carried_volume_parallel_enabled': False,
            'pickup_parallel_geometry_enabled': False,
            'empty_pickup_parallel_geometry_enabled': False,
            'pickup_post_retreat_parallel_geometry_enabled': False,
            'supported_compact_minimum_segment_seconds': .35,
            'scene_cartesian_wall_seconds': 420.0,
            'book_centered_place_height_above_point_m': 0.270,
            'book_centered_place_approach_x_m': 0.60,
            'book_centered_place_clearance_height_m': 0.40,
            'ik_position_tolerance': 0.012,
            'ik_orientation_tolerance': 0.10,
            'pick_ik_position_tolerance': 0.0005,
            'pick_ik_orientation_tolerance': 0.01,
            'place_joint_limit_margin_radians': 0.03,
            'cartesian_clearance_distance': 0.45,
            'top_row_cartesian_clearance_distance': 0.46,
            'cartesian_step': 0.06,
            'cartesian_joint_step_limit': 0.40,
            'pick_torso_height': 0.35,
            'lower_shelf_pick_enabled': True,
            'place_torso_height': 0.30,
            'measured_place_height_enabled': False,
            'empty_torso_planning_overlap_enabled': False,
            'settled_place_torso_skip_enabled': False,
            'settled_place_torso_retry_enabled': False,
            'preopen_stationary_enabled': False,
            'place_transition_stop_enabled': False,
            'release_only_clearance_timing_enabled': False,
            'place_finish_keep_torso_height_enabled': False,
            'gripper_settle_seconds': 1.0,
            'grasp_contact_max_age_seconds': 0.75,
            # Harmonic's Contact system publishes each active physics step
            # (500 Hz with official 2 ms steps), despite the sensor's 30 Hz tag.
            # The controller remains position-only; close with bounded position
            # steps and timestamped force feedback.
            'fine_gripper_close_enabled': False,
            'raw_contacts_enabled': False,
            'stock_gripper_close_diagnostic_enabled': False,
            'stock_gripper_close_target_position': 0.0,
            'fine_gripper_minimum_force': 1.0,
            'fine_gripper_step_wall_seconds': 5.0,
            'fine_gripper_wall_timeout_seconds': 300.0,
            'fine_gripper_micro_position_error_m': 0.000000025,
            'fine_gripper_micro_stationary_velocity_mps': 0.000002,
            'fine_gripper_micro_preload_steps': 20,
            'fine_gripper_micro_preload_step_m': 0.0000001,
            'fine_gripper_micro_motion_seconds': 0.20,
            'fine_gripper_micro_preload_limit_m': 0.000002,
            'adaptive_gripper_close_step': 0.001,
            'adaptive_gripper_step_motion_seconds': 0.32,
            'adaptive_gripper_step_settle_seconds': 0.10,
            'adaptive_gripper_preload_distance': 0.001,
            'adaptive_gripper_confirmation_seconds': 0.12,
            'adaptive_gripper_contact_force_minimum': 0.05,
            'adaptive_gripper_contact_force_maximum': 8.0,
            'adaptive_gripper_contact_samples': 3,
            'adaptive_gripper_contact_max_age_seconds': 0.15,
            'adaptive_gripper_contact_max_gap_seconds': 0.075,
            'adaptive_gripper_contact_min_span_seconds': 0.05,
            'adaptive_gripper_contact_max_skew_seconds': 0.05,
            'adaptive_gripper_velocity_tolerance': 0.003,
            'adaptive_gripper_effort_maximum': 4.0,
            'adaptive_gripper_effort_delta_maximum': 3.0,
            'adaptive_gripper_unilateral_travel_limit': 0.004,
            'adaptive_gripper_endpoint_tolerance': 0.0006,
            'adaptive_gripper_start_tolerance': 0.0015,
            # Official competition-book depth, width, and height.  The
            # additional padding is applied to every face during carried-path
            # checks rather than changing the empirical grasp depth.
            'carried_book_dimensions': [0.16, 0.02, 0.25],
            'carried_book_padding': 0.015,
            'carried_shelf_margin': 0.02,
            'carried_shelf_retreat_clearance_distance': 0.25,
            'carried_cradle_extension_distance': 0.12,
            'carried_cradle_retraction_distance': 0.05,
            'carried_cradle_roll_radians': -1.10,
            'carried_cradle_transfer_seconds': 0.75,
            'carried_supported_jaw_vertical_component': 0.75,
            'carried_transition_samples': 61,
            'carried_orientation_step_limit': 0.45,
            'carried_navigation_radius_limit': 0.45,
            'carried_maximum_tilt_radians': 1.0471975512,
        }
        for name, default in values.items():
            self.declare_parameter(name, default)

    def _publish_status(self, event: str, **fields) -> None:
        try:
            _add_planning_cpu_fields(self, event, fields)
        except Exception:
            pass
        message = String()
        message.data = encode_event(event, **fields)
        self.status_pub.publish(message)

    def _announce_ready(self) -> None:
        controllers_ready = all(
            client.server_is_ready()
            for client in (
                self.arm_client,
                self.right_arm_client,
                self.head_client,
                self.torso_client,
            )
        )
        gripper_relay_ready = self.gripper_pub.get_subscription_count() > 0
        joints_ready = all(
            name in self.joints
            for name in (*IK_JOINTS, *RIGHT_ARM_JOINTS, *HEAD_JOINTS)
        )
        if (
            self._ready_announced
            or not controllers_ready
            or not gripper_relay_ready
            or not joints_ready
        ):
            return
        self._ready_announced = True
        self._publish_status('ready', joints=len(self.joints))

    def _on_staging_odom(self, message: Odometry) -> None:
        if getattr(self, 'delivery_evidence_enabled', False):
            record_raw_odom(self, message)
        pose, twist = message.pose.pose, message.twist.twist
        sample = {
            'stamp_ns': _message_stamp_ns(message, int(self.get_clock().now().nanoseconds)),
            # New registered-scene admission requires raw producer metadata;
            # retain the legacy fallback field for its existing consumers.
            'producer_stamp_ns': (int(message.header.stamp.sec)*1_000_000_000
                                  + int(message.header.stamp.nanosec)),
            'frame_id': message.header.frame_id,
            'child_frame_id': message.child_frame_id,
            'pose': (float(pose.position.x), float(pose.position.y),
                     yaw_from_quaternion(pose.orientation)),
            'linear_speed': math.hypot(twist.linear.x, twist.linear.y),
            'angular_speed': abs(float(twist.angular.z)),
        }
        with self._lock:
            self._staging_odom = sample

    def _on_joint_state(self, message: JointState) -> None:
        if getattr(self, 'delivery_evidence_enabled', False):
            record_raw_joints(self, message)
        updates = {
            name: float(position)
            for name, position in zip(message.name, message.position)
        }
        feedback: Optional[GripperFeedback] = None
        try:
            gripper_index = list(message.name).index(
                'gripper_left_finger_joint'
            )
        except ValueError:
            gripper_index = -1
        if gripper_index >= 0 and gripper_index < len(message.position):
            position = float(message.position[gripper_index])
            velocity = (
                float(message.velocity[gripper_index])
                if gripper_index < len(message.velocity)
                else math.nan
            )
            effort = (
                float(message.effort[gripper_index])
                if gripper_index < len(message.effort)
                else math.nan
            )
            feedback = GripperFeedback(
                stamp_ns=_message_stamp_ns(
                    message,
                    int(self.get_clock().now().nanoseconds),
                ),
                position=position,
                velocity=velocity,
                effort=effort,
            )

        def apply_updates() -> None:
            self.joints.update(updates)
            velocities = getattr(self, '_joint_velocities', {})
            velocities.update({name: float(message.velocity[index])
                               if index < len(message.velocity) else math.nan
                               for index, name in enumerate(message.name)
                               if name in updates})
            self._joint_velocities = velocities
            stamps = getattr(self, '_joint_stamps_ns', {})
            stamp = _message_stamp_ns(message, int(self.get_clock().now().nanoseconds))
            stamps.update({name: stamp for name in updates})
            self._joint_stamps_ns = stamps
            if (_stock_diagnostic(self) and 'gripper_left_finger_joint' in updates
                    and (getattr(self, '_adaptive_close_active', False)
                         or getattr(self, '_held_book_corners', None) is not None)):
                values = (updates['gripper_left_finger_joint'], velocities.get('gripper_left_finger_joint', math.nan),
                          getattr(feedback, 'effort', math.nan))
                if not all(math.isfinite(value) for value in values):
                    if getattr(self, '_adaptive_close_active', False):
                        self._adaptive_overload_latched = self._adaptive_overload_latched or 'invalid_stock_joint_feedback'
                    if getattr(self, '_held_book_corners', None) is not None:
                        self._held_grip_sensor_fault = getattr(self, '_held_grip_sensor_fault', None) or 'invalid_stock_joint_feedback'
            if feedback is not None:
                samples = getattr(self, '_gripper_feedback_samples', None)
                if samples is None:
                    samples = deque(maxlen=128)
                    self._gripper_feedback_samples = samples
                samples.append(feedback)
                if _stock_diagnostic(self) and math.isfinite(feedback.effort):
                    self._stock_effort_peak = max(getattr(self, '_stock_effort_peak', 0.), abs(feedback.effort))
                if (
                    bool(getattr(self, '_adaptive_close_active', False))
                    and not _stock_diagnostic(self)
                    and math.isfinite(feedback.effort)
                    and getattr(self, '_adaptive_overload_latched', None)
                    is None
                ):
                    if abs(feedback.effort) > float(
                        getattr(self, 'adaptive_effort_maximum', math.inf)
                    ):
                        self._adaptive_overload_latched = 'effort_overload'
                    else:
                        baseline = float(
                            getattr(
                                self,
                                '_adaptive_effort_baseline_value',
                                math.nan,
                            )
                        )
                        if (
                            math.isfinite(baseline)
                            and abs(feedback.effort - baseline)
                            > float(
                                getattr(
                                    self,
                                    'adaptive_effort_delta_maximum',
                                    math.inf,
                                )
                            )
                        ):
                            self._adaptive_overload_latched = (
                                'effort_delta_overload'
                            )

        lock = getattr(self, '_lock', None)
        if lock is None:
            apply_updates()
        else:
            with lock:
                apply_updates()

        self._interrupt_adaptive_gripper_if_fault()

    _fine_contact_force_magnitude = staticmethod(_contact_force_magnitude)

    def _on_contacts(self, message: Contacts) -> None:
        _observe_place_contacts(self, message)
        now_ns = _message_stamp_ns(
            message,
            int(self.get_clock().now().nanoseconds),
            # Contacts can precede /clock delivery. Collapsing distinct physics
            # stamps would turn sequential forces into a false simultaneous sum.
            clamp_small_future=False,
        )
        lock = getattr(self, '_lock', None)
        if lock is None:
            target_model = getattr(self, '_target_book_model', None)
            contact_epoch = int(getattr(self, '_contact_epoch', 0))
        else:
            with lock:
                target_model = getattr(self, '_target_book_model', None)
                contact_epoch = int(getattr(self, '_contact_epoch', 0))
        robot_contact_models = set()
        external_hand_contact = False
        side_contacts: Dict[str, List[bool]] = {}
        side_forces: Dict[str, List[Dict[Tuple[str, str], float]]] = {}
        held_sensor_fault = None
        stock_models = set()
        for contact in getattr(message, 'contacts', []):
            first = getattr(getattr(contact, 'collision1', None), 'name', '')
            second = getattr(getattr(contact, 'collision2', None), 'name', '')
            pair = normalize_contact_pair(first, second)
            # Reuse only this delivered entry, including an invalid result.
            # Independent entries/messages still evaluate their own wrench.
            force_computed = False
            force_magnitude = None
            # Ignore contacts internal to the finger linkage. Any other
            # left-gripper contact prevents an empty-hand staging certificate.
            if ('gripper_left_' in first.lower()) != ('gripper_left_' in second.lower()):
                external_hand_contact = True
                # Target-only histories omit unrelated pairs. Preserve negative
                # raw observations without relabeling them as target force.
                raw_expected = target_model
                if _stock_diagnostic(self):
                    import re
                    other = second if 'gripper_left_' in first.lower() else first
                    models = [part for part in other.split('::') if re.fullmatch(
                        r'book_col_\d+_row_\d+_(red|green|yellow|blue)', part)]
                    if len(models) == 1 and models[0].endswith('_' + self.target_colour):
                        stock_models.add(models[0])
                        if raw_expected is None: raw_expected = models[0]
                raw_fault = held_contact_identity_fault(first, second, raw_expected)
                raw_force = _contact_force_magnitude(
                    contact, gripper_is_collision1='gripper_left_' in first.lower())
                force_magnitude = raw_force
                force_computed = True
                if raw_force is None:
                    raw_fault = raw_fault or 'invalid_held_contact_force'
                elif not _stock_diagnostic(self) and raw_force > float(getattr(self, 'adaptive_contact_force_maximum', math.inf)):
                    raw_fault = raw_fault or 'force_overload'
                held_sensor_fault = held_sensor_fault or raw_fault
            if is_target_book_non_gripper_robot_contact(
                pair,
                self.target_colour,
                None,
            ):
                observed_model = (
                    book_model_name(first) or book_model_name(second)
                )
                robot_contact_models.add(observed_model or '')
            if not is_intentional_gripper_book_contact(
                pair,
                self.target_colour,
                target_model,
            ):
                continue
            # The mission deliberately parks the right arm.  Contacts from its
            # gripper must never satisfy the left-arm pinch gate merely because
            # both grippers use the same fingertip-side naming convention.
            left_gripper_names = [
                name for name in pair if 'gripper_left_' in name.lower()
            ]
            if not left_gripper_names:
                continue
            observed_model = book_model_name(first) or book_model_name(second)
            model_key = observed_model or target_model or ''
            sides = side_contacts.setdefault(model_key, [False, False])
            forces = side_forces.setdefault(model_key, [{}, {}])
            if not force_computed:
                force_magnitude = _contact_force_magnitude(
                    contact,
                    gripper_is_collision1=(
                        'gripper_left_' in first.lower()
                    ),
                )
            for name in left_gripper_names:
                lowered = name.lower()
                if any(
                    token in lowered
                    for token in (
                        'base_finger_left',
                        'inner_finger_left',
                        'outer_finger_left',
                        'fingertip_left',
                    )
                ):
                    sides[0] = True
                    if force_magnitude is not None:
                        forces[0][pair] = max(
                            forces[0].get(pair, 0.0), force_magnitude,
                        )
                if any(
                    token in lowered
                    for token in (
                        'base_finger_right',
                        'inner_finger_right',
                        'outer_finger_right',
                        'fingertip_right',
                    )
                ):
                    sides[1] = True
                    if force_magnitude is not None:
                        forces[1][pair] = max(
                            forces[1].get(pair, 0.0), force_magnitude,
                        )

        newly_latched_model: Optional[str] = None

        def apply_updates() -> None:
            nonlocal newly_latched_model
            self._empty_hand_contact = (now_ns, external_hand_contact)
            if external_hand_contact:
                self._last_external_hand_contact_ns = now_ns
            if (external_hand_contact
                    and (getattr(self, '_empty_arm_motion_active', False)
                         or getattr(self, '_empty_arm_contact_guard', False))):
                new_empty_contact = not getattr(self, '_empty_arm_contact_latched', False)
                self._empty_arm_contact_latched = True
                # The action polling loop and gripper wait both observe this
                # interlock, even if the contact disappears on the next frame.
                self._cancel.set()
                if new_empty_contact:
                    self._publish_status('empty_arm_hazard', reason='external_hand_contact')
            selected_model = getattr(self, '_target_book_model', None)
            if _stock_diagnostic(self) and getattr(self, '_adaptive_close_active', False):
                fault = held_sensor_fault
                if len(stock_models) > 1 or (selected_model is not None and stock_models - {selected_model}):
                    fault = fault or 'unexpected_stock_contact_identity'
                if fault:
                    self._adaptive_overload_latched = self._adaptive_overload_latched or fault
                elif selected_model is None and len(stock_models) == 1:
                    selected_model = next(iter(stock_models))
                    self._target_book_model = selected_model
                    newly_latched_model = selected_model
            payload_under_control = bool(
                getattr(self, '_held_book_corners', None) is not None
                or getattr(self, '_transport_lock_engaged', False)
            )
            if (payload_under_control and selected_model is not None
                    and target_model == selected_model and held_sensor_fault):
                self._held_grip_sensor_fault = (
                    getattr(self, '_held_grip_sensor_fault', None) or held_sensor_fault)
            matched_held_robot_contact = bool(
                payload_under_control
                and robot_contact_models
                and (
                    selected_model is None
                    or '' in robot_contact_models
                    or (
                        selected_model in robot_contact_models
                    )
                )
            )
            if matched_held_robot_contact:
                # A fresh-retention probe changes the gripper-sample epoch, but
                # it must not erase collision evidence for a payload that is
                # still under control.  A successful release clears the held
                # state and transport lock under this same mutex, so callbacks
                # applied after release cannot re-latch an old collision.
                self._target_robot_contact_latched = True
            # Contact callbacks parse outside the lock.  A successful release
            # or a fresh-retention probe can reset the sample epoch meanwhile;
            # never let a pre-reset message repopulate the cleared state.
            if int(getattr(self, '_contact_epoch', 0)) != contact_epoch:
                if matched_held_robot_contact:
                    self._contact_generation = int(
                        getattr(self, '_contact_generation', 0)
                    ) + 1
                return
            samples = getattr(self, '_book_contact_samples', {})
            for model_key, (saw_left, saw_right) in side_contacts.items():
                timestamps = samples.setdefault(model_key, [0, 0])
                if saw_left:
                    timestamps[0] = max(timestamps[0], now_ns)
                if saw_right:
                    timestamps[1] = max(timestamps[1], now_ns)
            self._book_contact_samples = samples
            force_samples = getattr(
                self,
                '_book_contact_force_samples',
                {},
            )
            force_frames = getattr(self, '_book_contact_force_frames', {})
            for model_key, (left_forces, right_forces) in side_forces.items():
                if not left_forces and not right_forces:
                    continue
                histories = force_samples.get(model_key)
                if histories is None:
                    histories = [deque(maxlen=64), deque(maxlen=64)]
                    force_samples[model_key] = histories
                frame_sides = force_frames.setdefault(model_key, [{}, {}])

                def append_force(
                    side_index: int, values: Dict[Tuple[str, str], float],
                ) -> None:
                    if not values:
                        return
                    history = histories[side_index]
                    frames = frame_sides[side_index]
                    existing_frame = now_ns in frames
                    frame = frames.setdefault(now_ns, {})
                    # Gazebo reports all contact points for one collision pair
                    # together. Repeated reports of that same pair/stamp are
                    # duplicates, while different finger parts add real loads.
                    # Keep the larger duplicate to preserve the overload guard.
                    for pair_key, pair_force in values.items():
                        frame[pair_key] = max(frame.get(pair_key, 0.0), pair_force)
                    force = float(sum(frame.values()))
                    if _stock_diagnostic(self) and not math.isfinite(force):
                        if getattr(self, '_adaptive_close_active', False):
                            self._adaptive_overload_latched = self._adaptive_overload_latched or 'invalid_stock_contact_force'
                        if payload_under_control:
                            self._held_grip_sensor_fault = getattr(self, '_held_grip_sensor_fault', None) or 'invalid_stock_contact_force'
                    self._record_adaptive_force_peak(
                        model_key, side_index, force, now_ns,
                    )
                    if (not _stock_diagnostic(self) and payload_under_control and model_key == selected_model
                            and force > float(getattr(
                                self, 'adaptive_contact_force_maximum', math.inf))):
                        # Preserve the existing per-side cap across contact
                        # probes after acquisition, including distinct parts
                        # whose individual forces are each below that cap.
                        self._held_grip_sensor_fault = (
                            getattr(self, '_held_grip_sensor_fault', None) or 'force_overload')
                    if (
                        bool(getattr(self, '_adaptive_close_active', False))
                        and not _stock_diagnostic(self)
                        and force
                        > float(
                            getattr(
                                self,
                                'adaptive_contact_force_maximum',
                                math.inf,
                            )
                        )
                        and getattr(
                            self,
                            '_adaptive_overload_latched',
                            None,
                        )
                        is None
                    ):
                        self._adaptive_overload_latched = 'force_overload'
                    # The stores are created/reset together and history is
                    # sorted. Only its latest affected entry needs rebuilding
                    # for normal producer order; keep all fault/peak work above.
                    if getattr(history, "maxlen", None) == 64:
                        if (existing_frame and history
                                and now_ns == history[-1].stamp_ns
                                and len(frames) == len(history)):
                            history[-1] = ForceSample(now_ns, force)
                            return
                        if (not existing_frame
                                and len(frames) == len(history) + 1
                                and (not history or (
                                    now_ns > history[-1].stamp_ns
                                    and history[0].stamp_ns in frames))):
                            if len(history) == 64:
                                del frames[history[0].stamp_ns]
                            history.append(ForceSample(now_ns, force))
                            return
                    # Different contact publishers can arrive out of order.
                    # A late frame must not erase newer measurements. Explicit
                    # contact-epoch resets clear both stores together.
                    stamps = sorted(frames)
                    for old_stamp in stamps[:-64]:
                        del frames[old_stamp]
                    history.clear()
                    history.extend(
                        ForceSample(stamp, float(sum(frames[stamp].values())))
                        for stamp in stamps[-64:]
                    )

                if left_forces:
                    append_force(0, left_forces)
                if right_forces:
                    append_force(1, right_forces)
            self._book_contact_force_samples = force_samples
            self._book_contact_force_frames = force_frames

            selected_model = getattr(self, '_target_book_model', None)
            if selected_model is None:
                maximum_age_ns = int(
                    float(getattr(self, 'grasp_contact_max_age', 0.75)) * 1e9
                )
                candidates = [
                    (min(timestamps), model)
                    for model, timestamps in samples.items()
                    if model
                    and all(
                        timestamp > 0
                        and -100_000_000 <= now_ns - timestamp <= maximum_age_ns
                        for timestamp in timestamps
                    )
                ]
                if candidates:
                    _, selected_model = max(candidates)
                    self._target_book_model = selected_model
                    newly_latched_model = selected_model

            if selected_model is not None:
                selected = samples.get(selected_model, samples.get('', [0, 0]))
                self._left_target_contact_ns = selected[0]
                self._right_target_contact_ns = selected[1]
            elif '' in samples:
                self._left_target_contact_ns = samples[''][0]
                self._right_target_contact_ns = samples[''][1]

            anonymous_contact_is_held = bool(
                selected_model is None
                and '' in samples
                and all(
                    timestamp > 0
                    and -100_000_000 <= now_ns - timestamp
                    <= int(
                        float(getattr(self, 'grasp_contact_max_age', 0.75))
                        * 1e9
                    )
                    for timestamp in samples['']
                )
            )
            matched_robot_contact = bool(
                robot_contact_models
                and (
                    (
                        selected_model is not None
                        and (
                            selected_model in robot_contact_models
                            or '' in robot_contact_models
                        )
                    )
                    or anonymous_contact_is_held
                )
            )
            if matched_robot_contact:
                # Keep the collision recorded even if the contact bridge stops
                # publishing it before the trajectory watchdog next polls.
                self._target_robot_contact_latched = True

            if matched_robot_contact or side_contacts:
                self._contact_generation = int(
                    getattr(self, '_contact_generation', 0)
                ) + 1

            # Publish while the same lock still protects the identity.  A
            # successful open therefore cannot reset the identity before its
            # latch event is emitted.
            if newly_latched_model is not None:
                publisher = getattr(self, '_publish_status', None)
                if callable(publisher):
                    publisher(
                        'target_book_latched',
                        model=newly_latched_model,
                    )

        if lock is None:
            apply_updates()
        else:
            with lock:
                apply_updates()

        self._interrupt_adaptive_gripper_if_fault()
        controller = getattr(self, '_fine_gripper_controller', None)
        if controller is not None:
            try:
                controller.observe_contacts(message)
            except Exception as exc:
                controller._hold('fine_contact_callback_failed')
                self._publish_status('fine_gripper_progress',
                                     stage='contact_callback_error', detail=str(exc))

    def _on_book(self, message: PointStamped) -> None:
        self.latest_book = message

    def _on_target_tracking(self, message: PointCloud) -> None:
        """Cache only the latest volatile observation message."""
        lock = getattr(self, '_lock', None)
        if lock is None:
            self._latest_target_tracking = message
        else:
            with lock:
                self._latest_target_tracking = message

    def _on_target_tracking_status(self, message: String) -> None:
        """Apply availability edges and invalidate cached vision immediately."""
        status = _decode_target_tracking_status(message.data, self.target_colour)
        lock = getattr(self, '_lock', None)

        def update() -> None:
            self._target_tracking_status = status
            if not status.available:
                # An occlusion/failure edge must make the last geometrically
                # good cloud unusable.  Reacquisition publishes a new identity.
                self._latest_target_tracking = None

        if lock is None:
            update()
        else:
            with lock:
                update()

    def _tracking_observation_from_cloud(
        self,
        cloud: PointCloud,
        status: _TargetTrackingStatus,
    ) -> TargetBookObservation:
        """Validate the complete wire contract before any TF lookup."""
        if not status.available:
            raise RuntimeError(
                f'target tracking unavailable ({status.reason})'
            )
        frame_id = str(cloud.header.frame_id).strip()
        seconds = int(cloud.header.stamp.sec)
        nanoseconds = int(cloud.header.stamp.nanosec)
        if not frame_id:
            raise RuntimeError('target observation frame is empty')
        if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
            raise RuntimeError('target observation timestamp is invalid')
        stamp_ns = seconds * 1_000_000_000 + nanoseconds
        if stamp_ns <= 0:
            raise RuntimeError('target observation timestamp is unset')

        points = np.asarray(
            [[point.x, point.y, point.z] for point in cloud.points],
            dtype=float,
        )
        if points.shape != (5, 3) or not np.all(np.isfinite(points)):
            raise RuntimeError(
                'target observation must contain exactly five finite points'
            )
        try:
            roles = _point_cloud_channel(
                cloud,
                'point_role',
                repeated=False,
            )
            if not np.allclose(
                roles,
                np.arange(5, dtype=float),
                rtol=0.0,
                atol=1e-4,
            ):
                raise ValueError('point roles are not center then ordered corners')

            row = _integer_metadata(
                _point_cloud_channel(cloud, 'target_row')[0],
                'target_row',
                minimum=1,
            )
            generation = _integer_metadata(
                _point_cloud_channel(cloud, 'track_generation')[0],
                'track_generation',
                minimum=1,
            )
            sequence = _integer_metadata(
                _point_cloud_channel(cloud, 'sequence')[0],
                'sequence',
                minimum=1,
            )

            def scalar(name: str) -> float:
                return float(_point_cloud_channel(cloud, name)[0])

            detection_confidence = scalar('detection_confidence')
            quality_confidence = scalar('quality_confidence')
            identity_confidence = scalar('identity_continuity_confidence')
            depth_coverage = scalar('depth_coverage')
            plane_residual = scalar('plane_residual_m')
            center_uncertainty = scalar('center_uncertainty_m')
            extent_uncertainty = scalar('extent_uncertainty_m')
            orientation_uncertainty = scalar('orientation_uncertainty_rad')
            short_extent = scalar('short_extent_m')
            long_extent = scalar('long_extent_m')
            observed_rate = scalar('observed_rate_hz')
            face_normal = np.asarray(
                [
                    scalar('face_normal_x'),
                    scalar('face_normal_y'),
                    scalar('face_normal_z'),
                ],
                dtype=float,
            )
            long_axis = np.asarray(
                [
                    scalar('long_axis_x'),
                    scalar('long_axis_y'),
                    scalar('long_axis_z'),
                ],
                dtype=float,
            )
        except ValueError as exc:
            raise RuntimeError(f'invalid target observation: {exc}') from exc

        if row > 4 or row != status.row:
            raise RuntimeError('target observation row does not match status')
        if generation != status.track_generation:
            raise RuntimeError(
                'target observation generation does not match status'
            )
        if stamp_ns < status.stamp_ns or sequence < status.sequence:
            raise RuntimeError('target observation predates its valid status')
        if not (
            0.0 < detection_confidence <= 1.0
            and 0.0 < quality_confidence <= 1.0
            and 0.0 < identity_confidence <= 1.0
        ):
            raise RuntimeError('target confidence metadata is invalid')
        # Channels use float32.  Admit only their representation-scale roundoff
        # at configured boundaries, never a physically meaningful relaxation.

        def upper_bound(value: float, limit: float) -> bool:
            tolerance = max(1e-9, 2e-6 * abs(limit))
            return 0.0 <= value <= limit + tolerance

        coverage_tolerance = max(
            1e-9,
            2e-6 * self.target_tracking_minimum_depth_coverage,
        )
        if not (
            depth_coverage + coverage_tolerance
            >= self.target_tracking_minimum_depth_coverage
            and depth_coverage <= 1.0
        ):
            raise RuntimeError('target depth coverage is insufficient')
        if not upper_bound(
            plane_residual,
            self.target_tracking_maximum_plane_residual,
        ):
            raise RuntimeError('target plane residual exceeds the safe bound')
        if not upper_bound(
            center_uncertainty,
            self.target_tracking_maximum_center_uncertainty,
        ):
            raise RuntimeError('target center uncertainty exceeds the safe bound')
        if not upper_bound(
            extent_uncertainty,
            self.target_tracking_maximum_extent_uncertainty,
        ):
            raise RuntimeError('target extent uncertainty exceeds the safe bound')
        if not upper_bound(
            orientation_uncertainty,
            self.target_tracking_maximum_orientation_uncertainty,
        ):
            raise RuntimeError(
                'target orientation uncertainty exceeds the safe bound'
            )
        if observed_rate + 1e-6 < self.target_tracking_minimum_rate:
            raise RuntimeError('target observation rate is below the safe bound')
        if not 0.0 < short_extent <= long_extent:
            raise RuntimeError('target face extents are invalid')

        normal_magnitude = float(np.linalg.norm(face_normal))
        axis_magnitude = float(np.linalg.norm(long_axis))
        if (
            abs(normal_magnitude - 1.0) > 0.02
            or abs(axis_magnitude - 1.0) > 0.02
        ):
            raise RuntimeError('target orientation vectors are not unit length')
        face_normal /= normal_magnitude
        long_axis /= axis_magnitude
        if abs(float(np.dot(face_normal, long_axis))) > 0.05:
            raise RuntimeError('target orientation vectors are not orthogonal')

        corners = points[1:]
        horizontal = (
            (corners[1] + corners[2]) - (corners[0] + corners[3])
        ) * 0.5
        vertical = (
            (corners[2] + corners[3]) - (corners[0] + corners[1])
        ) * 0.5
        horizontal_extent = float(np.linalg.norm(horizontal))
        vertical_extent = float(np.linalg.norm(vertical))
        measured_short = min(horizontal_extent, vertical_extent)
        measured_long = max(horizontal_extent, vertical_extent)
        # Both values originate from these same corners, so only Point32 /
        # ChannelFloat32 serialization roundoff is admissible here.
        extent_tolerance = max(2e-6, 5e-5 * long_extent)
        if (
            min(horizontal_extent, vertical_extent) <= 1e-5
            or abs(measured_short - short_extent) > extent_tolerance
            or abs(measured_long - long_extent) > extent_tolerance
        ):
            raise RuntimeError('target corners do not match reported extents')
        derived_long_axis = (
            vertical / vertical_extent
            if vertical_extent >= horizontal_extent
            else horizontal / horizontal_extent
        )
        if float(np.dot(derived_long_axis, long_axis)) < 0.995:
            raise RuntimeError('target corners do not match reported long axis')
        derived_normal = -np.cross(horizontal, vertical)
        derived_normal_magnitude = float(np.linalg.norm(derived_normal))
        if derived_normal_magnitude <= 1e-9:
            raise RuntimeError('target corners are geometrically degenerate')
        derived_normal /= derived_normal_magnitude
        if float(np.dot(derived_normal, face_normal)) < 0.995:
            raise RuntimeError('target corners do not match reported face normal')

        return TargetBookObservation(
            stamp_ns=stamp_ns,
            frame_id=frame_id,
            target_colour=status.target_colour,
            row=row,
            center=tuple(float(value) for value in points[0]),
            corners=tuple(
                tuple(float(value) for value in corner)
                for corner in corners
            ),
            face_normal=tuple(float(value) for value in face_normal),
            long_axis=tuple(float(value) for value in long_axis),
            short_extent_m=short_extent,
            long_extent_m=long_extent,
            # Image-space association already ran inside the producer.  The
            # transport contract intentionally contains only metric geometry.
            bbox=(0, 0, 0, 0),
            image_center=(0.0, 0.0),
            detection_confidence=detection_confidence,
            depth_coverage=depth_coverage,
            plane_residual_m=plane_residual,
            center_uncertainty_m=center_uncertainty,
            extent_uncertainty_m=extent_uncertainty,
            orientation_uncertainty_rad=orientation_uncertainty,
            quality_confidence=quality_confidence,
            track_id=status.track_id,
            track_generation=generation,
            sequence=sequence,
            interframe_period_s=1.0 / observed_rate,
            observed_rate_hz=observed_rate,
            identity_continuity_confidence=identity_confidence,
        )

    def _transform_target_observation(
        self,
        observation: TargetBookObservation,
        target_frame: str,
    ) -> TargetBookObservation:
        """Transform all five metric points at their measurement timestamp."""
        target_frame = str(target_frame).strip()
        if not target_frame:
            raise RuntimeError('target observation output frame is empty')
        rotation = np.eye(3, dtype=float)
        translation = np.zeros(3, dtype=float)
        if observation.frame_id != target_frame:
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    observation.frame_id,
                    Time(
                        nanoseconds=observation.stamp_ns,
                        clock_type=ClockType.ROS_TIME,
                    ),
                    timeout=Duration(seconds=self.tf_timeout),
                ).transform
                rotation = _quaternion_rotation_matrix(transform.rotation)
                translation = np.asarray(
                    [
                        transform.translation.x,
                        transform.translation.y,
                        transform.translation.z,
                    ],
                    dtype=float,
                )
            except (TransformException, ValueError) as exc:
                raise RuntimeError(
                    'cannot transform live target observation '
                    f'to {target_frame}: {exc}'
                ) from exc
            if not np.all(np.isfinite(translation)):
                raise RuntimeError('target-tracking transform is not finite')

        source_points = np.asarray(
            [observation.center, *observation.corners],
            dtype=float,
        )
        transformed = source_points @ rotation.T + translation
        transformed_normal = rotation @ np.asarray(
            observation.face_normal,
            dtype=float,
        )
        transformed_axis = rotation @ np.asarray(
            observation.long_axis,
            dtype=float,
        )
        return TargetBookObservation(
            stamp_ns=observation.stamp_ns,
            frame_id=target_frame,
            target_colour=observation.target_colour,
            row=observation.row,
            center=tuple(float(value) for value in transformed[0]),
            corners=tuple(
                tuple(float(value) for value in corner)
                for corner in transformed[1:]
            ),
            face_normal=tuple(float(value) for value in transformed_normal),
            long_axis=tuple(float(value) for value in transformed_axis),
            short_extent_m=observation.short_extent_m,
            long_extent_m=observation.long_extent_m,
            bbox=observation.bbox,
            image_center=observation.image_center,
            detection_confidence=observation.detection_confidence,
            depth_coverage=observation.depth_coverage,
            plane_residual_m=observation.plane_residual_m,
            center_uncertainty_m=observation.center_uncertainty_m,
            extent_uncertainty_m=observation.extent_uncertainty_m,
            orientation_uncertainty_rad=(
                observation.orientation_uncertainty_rad
            ),
            quality_confidence=observation.quality_confidence,
            track_id=observation.track_id,
            track_generation=observation.track_generation,
            sequence=observation.sequence,
            interframe_period_s=observation.interframe_period_s,
            observed_rate_hz=observation.observed_rate_hz,
            identity_continuity_confidence=(
                observation.identity_continuity_confidence
            ),
        )

    def _current_target_observation(
        self,
        *,
        target_frame: str = 'base_footprint',
        maximum_age_seconds: Optional[float] = None,
    ) -> TargetBookObservation:
        """Return one fresh, status-authorized RGB-D observation or raise."""
        maximum_age = self.target_tracking_max_age
        if maximum_age_seconds is not None:
            requested_age = float(maximum_age_seconds)
            if not math.isfinite(requested_age) or requested_age <= 0.0:
                raise ValueError('maximum target-observation age must be positive')
            maximum_age = min(maximum_age, requested_age)
        maximum_age = min(0.15, maximum_age)

        lock = getattr(self, '_lock', None)
        if lock is None:
            cloud = getattr(self, '_latest_target_tracking', None)
            status = getattr(
                self,
                '_target_tracking_status',
                _TargetTrackingStatus(False, 'tracking_status_not_received'),
            )
        else:
            with lock:
                cloud = self._latest_target_tracking
                status = self._target_tracking_status
        if not status.available:
            raise RuntimeError(f'target tracking unavailable ({status.reason})')
        if cloud is None:
            raise RuntimeError('no live target observation is available')

        source = self._tracking_observation_from_cloud(cloud, status)

        def checked_age() -> None:
            age = (
                self.get_clock().now().nanoseconds - source.stamp_ns
            ) / 1e9
            if age < -0.02:
                raise RuntimeError(
                    f'target observation is from the future (age={age:.3f}s)'
                )
            if age > maximum_age:
                raise RuntimeError(
                    f'target observation is stale (age={age:.3f}s)'
                )

        # Check both before and after the potentially blocking exact-time TF.
        checked_age()
        transformed = self._transform_target_observation(source, target_frame)
        checked_age()

        # An occlusion edge can arrive while TF waits.  It invalidates even a
        # cloud that was fresh when this method started.
        if lock is None:
            current_status = self._target_tracking_status
        else:
            with lock:
                current_status = self._target_tracking_status
        if current_status != status or not current_status.available:
            raise RuntimeError('target tracking changed during TF lookup')
        return transformed

    def _guard_target_motion(
        self,
        reference: TargetBookObservation,
        *,
        maximum_center_motion_m: float = 0.001,
        maximum_corner_motion_m: Optional[float] = None,
        maximum_orientation_change_rad: Optional[float] = None,
    ) -> TargetTrackingResult:
        """Wait for one newer frame, then prove bounded same-target motion."""
        started_ns = self.get_clock().now().nanoseconds
        # Use ROS time for the safety interval so a slower-than-real-time
        # simulation has the same semantics.  A separate wall-clock failsafe
        # still bounds a paused/broken clock.
        wall_deadline = time.monotonic() + max(
            2.0,
            10.0 * self.target_tracking_guard_wait,
        )
        while not self._cancel.is_set():
            current = self._current_target_observation(
                target_frame=reference.frame_id,
            )
            if (
                current.track_id != reference.track_id
                or current.track_generation != reference.track_generation
                or current.target_colour != reference.target_colour
                or current.row != reference.row
            ):
                raise RuntimeError('target identity changed during guarded motion')
            if (
                current.stamp_ns > reference.stamp_ns
                and current.sequence > reference.sequence
            ):
                result = guard_target_motion(
                    reference,
                    current,
                    maximum_center_motion_m=maximum_center_motion_m,
                    maximum_corner_motion_m=maximum_corner_motion_m,
                    maximum_orientation_change_rad=(
                        maximum_orientation_change_rad
                    ),
                )
                if not result.ok:
                    raise RuntimeError(
                        f'target motion guard failed ({result.reason})'
                    )
                return result
            now_ns = self.get_clock().now().nanoseconds
            elapsed = (now_ns - started_ns) / 1e9
            if elapsed < -0.02:
                raise RuntimeError('clock moved backward during target guard')
            if (
                elapsed >= self.target_tracking_guard_wait
                or time.monotonic() >= wall_deadline
            ):
                raise RuntimeError('no newer target observation arrived')
            time.sleep(0.01)
        raise RuntimeError('target motion guard cancelled')

    def _on_bin(self, message: PointStamped) -> None:
        if bin_message_is_current(
            message, self.get_clock().now().nanoseconds,
            getattr(self, 'bin_invalidated_ns', -1),
        ):
            self.bin_candidate = message
            self.latest_bin = message if bin_message_is_verified(
                message, self.get_clock().now().nanoseconds,
                getattr(self, 'bin_verified_ns', -1),
                getattr(self, 'bin_invalidated_ns', -1),
            ) else None

    def _on_bin_status(self, message: String) -> None:
        payload = decode_event(message.data)
        from .place_input_handoff import observe_idle
        observe_idle(self, payload)
        if payload.get('mode') == 'bin' and 'bin_valid' in payload:
            if getattr(self, 'table_scene_required', False):
                previous = getattr(self, '_latest_table_scene_entry', {})
                if int(payload.get('observation_stamp_ns', -1)) >= previous.get('observation_stamp_ns', -1):
                    self._latest_table_scene_entry = payload
            attribute = 'bin_verified_ns' if payload['bin_valid'] is True else 'bin_invalidated_ns'
            setattr(self, attribute, max(
                getattr(self, attribute, -1), int(payload.get('observation_stamp_ns', -1)),
            ))
            candidate = getattr(self, 'bin_candidate', None)
            self.latest_bin = candidate if bin_message_is_verified(
                candidate, self.get_clock().now().nanoseconds,
                getattr(self, 'bin_verified_ns', -1),
                getattr(self, 'bin_invalidated_ns', -1),
            ) else None

    def _payload_hazard_reason(
        self,
        *,
        max_age: float = 0.20,
    ) -> Optional[str]:
        contact_fault = _place_contact_fault(self)
        if contact_fault is not None:
            return contact_fault
        scene_reference = getattr(self, '_active_place_scene_reference', None)
        if scene_reference is not None:
            try:
                measured_scene_context(self, scene_reference)
            except RuntimeError as exc:
                return str(exc)
        if _stock_diagnostic(self):
            raw_fault = getattr(self, '_held_grip_sensor_fault', None)
            if raw_fault: return str(raw_fault)
            feedback = self._latest_gripper_feedback()
            now = int(self.get_clock().now().nanoseconds)
            if (feedback is None or not all(math.isfinite(v) for v in
                    (feedback.position, feedback.velocity, feedback.effort))
                    or not -100_000_000 <= now-feedback.stamp_ns <= 150_000_000):
                return 'invalid_stock_retained_feedback'
        latched = getattr(self, '_payload_hazard_latched', None)
        if latched is not None:
            return str(latched)
        if bool(getattr(self, '_target_robot_contact_latched', False)):
            return 'payload_robot_contact'
        if not self._target_contact_recent(max_age=max_age):
            return 'contact_lost'
        return None

    def _monitor_held_payload(self) -> None:
        """Latch and report a carried-payload hazard outside arm actions."""
        if getattr(self, '_head_return_state', None) is not None:
            from .head_return_overlap import monitor
            monitor(self)
        lock = getattr(self, '_lock', None)

        def monitoring_is_suppressed() -> bool:
            return bool(
                not getattr(self, '_payload_monitor_enabled', False)
                or getattr(self, '_retention_probe_active', False)
                or getattr(self, '_held_book_corners', None) is None
                or getattr(self, '_payload_hazard_latched', None) is not None
            )

        if lock is None:
            if monitoring_is_suppressed():
                return
            contact_generation = int(getattr(self, '_contact_generation', 0))
        else:
            with lock:
                if monitoring_is_suppressed():
                    return
                contact_generation = int(
                    getattr(self, '_contact_generation', 0)
                )
        reason = self._payload_hazard_reason(max_age=0.20)
        if reason is None:
            return
        # Suppression can begin while the contact snapshot above is being
        # evaluated (for example, an intentional release or a fresh probe).
        # Recheck under the same lock used to enable suppression before
        # committing an irreversible hazard latch.
        if lock is None:
            if (
                monitoring_is_suppressed()
                or int(getattr(self, '_contact_generation', 0))
                != contact_generation
            ):
                return
            self._payload_hazard_latched = reason
        else:
            with lock:
                if (
                    monitoring_is_suppressed()
                    or int(getattr(self, '_contact_generation', 0))
                    != contact_generation
                ):
                    return
                self._payload_hazard_latched = reason
        if _stock_diagnostic(self): self._hold_adaptive_gripper(str(reason))
        self._publish_status('payload_hazard', reason=reason)

    def _on_command(self, message: String) -> None:
        payload = decode_event(message.data)
        command = str(payload.get('event', message.data)).strip().lower()
        correlation = ({key: payload.get(key) for key in
            ('trial_id', 'placement_attempt_id', 'target_model')}
            if command == 'place' and getattr(self, 'delivery_evidence_enabled', False)
            else {})
        if command == 'place':
            from .place_input_handoff import command_fields
            correlation.update(command_fields(payload))
        if command == 'preposition_bin_head' or (payload or {}).get('head_return_id'):
            correlation.update({key: (payload or {}).get(key) for key in ('trial_id', 'head_return_id', 'target_model')})
        if 'empty_head_timing_id' in payload:
            correlation.update({k:payload.get(k) for k in ('trial_id', 'empty_head_timing_id')})
        if command in ('cancel', 'abort', 'stop'):
            if (getattr(self, '_release_pose_owner', None) is not None
                    and release_pose_finish.cancel_postopen(self)):
                return
            self._empty_arm_preparation = None
            self._cancel.set()
            with self._lock:
                goal_handles = list(self._goal_handles)
            for goal_handle in goal_handles:
                try:
                    goal_handle.cancel_goal_async()
                except Exception:
                    pass
            if _stock_diagnostic(self) and (getattr(self, '_adaptive_close_active', False)
                    or getattr(self, '_held_book_corners', None) is not None):
                self._hold_adaptive_gripper('cancelled')
            self._publish_status('cancelled')
            return
        with self._lock:
            if getattr(self, '_initial_stow_serial_owner', None) is not None:
                self._publish_status('rejected', command=command,
                    reason='initial_stow_serial_owner_active', **correlation)
                return
            if getattr(self, '_empty_torso_planning_owner', None) is not None:
                self._publish_status('rejected', command=command,
                    reason='empty_torso_planning_owner_active', **correlation)
                return
            if (getattr(self, '_release_pose_owner', None) is not None
                    or getattr(self, '_release_pose_fault_latched', None) is not None):
                self._publish_status('rejected', command=command,
                    reason='release_pose_owner_or_fault_active', **correlation)
                return
            head_owner = getattr(self, '_head_return_state', None)
            if head_owner is not None and not (
                    command == 'look_bin' and all(payload.get(key) == value
                    for key, value in head_owner['identity'].items())):
                self._publish_status('rejected', command=command,
                    reason='head_return_owner_active', **correlation)
                return
        with self._lock:
            if getattr(self, '_empty_head_timing_owner', None) is not None:
                self._publish_status('rejected', command=command,
                    reason='empty_head_timing_owner_active', **correlation)
                return
            if getattr(self, '_raw_contacts_first_failure', None) is not None:
                self._publish_status(
                    'rejected', command=command,
                    reason='raw_contacts_decode_failed', **correlation,
                )
                return
            if self._busy:
                self._publish_status('rejected', command=command, reason='busy', **correlation)
                return
            if getattr(self, '_pending_retained_acceptances', ()):
                self._publish_status(
                    'rejected', command=command,
                    reason='retained_goal_acceptance_unresolved', **correlation,
                )
                return
            if self._goal_handles:
                # A handle remains registered while a controller goal is
                # active, including when cancellation could not be confirmed.
                # Do not clear the cancel interlock and issue a conflicting
                # command from that unknown physical state.
                self._publish_status(
                    'rejected',
                    command=command,
                    reason='controller_goal_unresolved', **correlation,
                )
                return
            self._busy = True
            self._cancel.clear()
        _cpu_command_target = self._run_command
        try:
            _cpu_command_target = _planning_cpu_command_target(self, _cpu_command_target)
        except Exception:
            pass
        threading.Thread(
            target=_cpu_command_target,
            args=(command, dict(payload)),
            name=f'erc-manipulation-{command}',
            daemon=True,
        ).start()

    def _run_command(self, command: str, payload: Optional[Dict] = None) -> None:
        correlation = ({key: (payload or {}).get(key) for key in
            ('trial_id', 'placement_attempt_id', 'target_model')}
            if command == 'place' and getattr(self, 'delivery_evidence_enabled', False)
            else {})
        if command == 'place':
            from .place_input_handoff import command_fields
            correlation.update(command_fields(payload))
        if command == 'preposition_bin_head' or (payload or {}).get('head_return_id'):
            correlation.update({key: (payload or {}).get(key) for key in ('trial_id', 'head_return_id', 'target_model')})
        if _empty_head_timing.ID in (payload or {}):
            correlation.update({k:payload.get(k) for k in ('trial_id', _empty_head_timing.ID)})
        self._publish_status('started', command=command, **correlation)
        try:
            initial_stow = (_initial_stow_serial.begin_command(self, command)
                if getattr(self, 'initial_stow_serial_timing_enabled', False) else None)
            if command == 'pick':
                self._empty_head_pick_started = True
            empty_head = _empty_head_timing.prepare(self, command, payload)
            if command == 'prepare_pick_approach':
                from .empty_arm_preparation import prepare_pick_approach
                preparation_handler = lambda: prepare_pick_approach(self, payload or {})
            else:
                preparation_handler = None
            handlers = {
                'stow': (lambda: self._stow(initial_stow_serial=initial_stow))
                    if initial_stow is not None else self._stow,
                'look_markers': lambda: self._move_head(0.0, 0.20,
                    **({'empty_head_timing': empty_head} if empty_head is not None else {})),
                'look_books': lambda: self._move_head(0.0, self.book_overview_tilt,
                    **({'empty_head_timing': empty_head} if empty_head is not None else {})),
                'look_bin': lambda: self._move_head(0.0, -0.60),
                'pick': lambda: self._pick(payload) if payload else self._pick(),
                'compact_transport': self._compact_transport,
                'place': (lambda: self._place(payload))
                    if (getattr(self, 'delivery_evidence_enabled', False)
                        or 'place_input_request_id' in (payload or {})) else self._place,
                'open_gripper': self._open_gripper,
                'prepare_pick_approach': preparation_handler,
            }
            for row, tilt in enumerate(self.book_row_tilts, start=1):
                handlers[f'look_book_row_{row}'] = (
                    lambda row_tilt=tilt: self._move_head(0.0, row_tilt,
                        **({'empty_head_timing': empty_head} if empty_head is not None else {}))
                )
            if command == 'preposition_bin_head':
                from .head_return_overlap import run
                handlers[command] = lambda: run(self, payload or {})
            elif command == 'look_bin' and (payload or {}).get('head_return_id'):
                from .head_return_overlap import finish
                handlers[command] = lambda: finish(self, payload or {})
            handler = handlers.get(command)
            if handler is None:
                raise ValueError(f'unsupported command: {command}')
            if self.dry_run:
                time.sleep(0.05)
                result = True
            else:
                result = bool(handler())
            if self._cancel.is_set():
                self._publish_status('cancelled', command=command, **correlation)
            elif result:
                fields = (
                    dict(getattr(self, '_empty_arm_preparation', None) or {})
                    if command == 'prepare_pick_approach' else {}
                )
                if empty_head is not None:
                    fields.update({k:v for k,v in empty_head.completion.items() if k not in correlation})
                if (command == 'place' and getattr(self, '_release_pose_owner', None) is not None
                        and release_pose_finish.publish_terminal(self, correlation)):
                    pass
                else:
                    self._publish_status('succeeded', command=command, **fields, **correlation)
            else:
                self._publish_status('failed', command=command, reason='execution_failed', **correlation)
        except Exception as exc:
            # RcutilsLogger in ROS 2 Humble has no ``exception`` method.  Log
            # through the supported API so the failure status is still sent.
            self.get_logger().error(f'{command} failed: {exc}')
            failure_fields = {}
            if (command == 'place'
                    and isinstance(exc, release_pose_finish.ReleasePoseFeedbackStale)):
                failure_fields['release_pose_feedback'] = exc.details
            try:
                # This is the existing outside-lock failed event. Diagnostics
                # do not publish from a sensor callback or change its reason.
                self._publish_status('failed', command=command, reason=str(exc),
                                     **failure_fields, **correlation)
            except Exception as publication_error:
                if failure_fields:
                    # Preserve the original safety fault if its extended
                    # failure status cannot be serialized or delivered.
                    raise exc from publication_error
                raise
        finally:
            if command == 'place':
                _deactivate_place_contacts(self)
                if getattr(self, '_release_pose_owner', None) is not None:
                    release_pose_finish.deactivate(self)
            with self._lock:
                if command == 'place':
                    self._active_place_scene_reference = None
                if command == 'prepare_pick_approach':
                    self._empty_arm_motion_active = False
                self._busy = False

    @staticmethod
    def _wait_future(future, timeout: float):
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            raise TimeoutError('ROS action future timed out')
        return future.result()

    def _cancel_goal_and_confirm(self, goal_handle, result_future) -> bool:
        """Request cancellation and wait briefly for a terminal result."""
        try:
            cancel_future = goal_handle.cancel_goal_async()
            self._wait_future(cancel_future, min(2.0, self.timeout))
        except Exception:
            # Another cancellation request can still make the goal terminal.
            pass
        terminal_deadline = time.monotonic() + min(5.0, self.timeout)
        while not result_future.done() and time.monotonic() < terminal_deadline:
            time.sleep(0.02)
        return result_future.done()

    @track_completed_torso_hold
    def _follow(
        self,
        client: ActionClient,
        names: Sequence[str],
        positions: Sequence[float],
        duration: float,
        *,
        head_preflight: bool = False,
        empty_head_owner=None,
        pre_send_check=None,
        trajectory_duration=None,
        _torso_hold_token=None,
        empty_pickup_setup=False,
        empty_torso_owner=None,
        initial_stow_owner=None,
    ) -> bool:
        if initial_stow_owner is not None:
            if (empty_torso_owner is not None or empty_head_owner is not None
                    or head_preflight or empty_pickup_setup or pre_send_check is not None
                    or trajectory_duration is not None):
                raise ValueError('initial stow timing cannot combine with another motion option')
            initial_stow_owner.follow_started(client, names, positions, duration, _torso_hold_token)
        if empty_torso_owner is not None and empty_head_owner is not None:
            raise ValueError('empty torso and head owners are mutually exclusive')
        if empty_torso_owner is not None:
            empty_torso_owner.follow_started(client, names, positions, duration, _torso_hold_token)
        if self._cancel.is_set():
            return False
        _require_place_contact_clear(self)
        scene_reference = getattr(self, '_active_place_scene_reference', None)
        if scene_reference is not None:
            _require_place_contact_clear(self)
            measured_scene_context(self, scene_reference)
        with self._lock:
            payload_robot_contact = bool(
                getattr(self, '_payload_robot_watchdog_enabled', False)
                and getattr(self, '_target_robot_contact_latched', False)
            )
            if payload_robot_contact:
                self._payload_hazard_latched = 'payload_robot_contact'
        if payload_robot_contact:
            self._publish_status(
                'payload_hazard',
                reason='payload_robot_contact',
            )
            return False
        if head_preflight:
            if client is not self.head_client or tuple(names) != tuple(HEAD_JOINTS):
                raise ValueError('head preflight requires the head controller and joints')
            completed = getattr(self, '_last_completed_head_target', None)
            # A failed, cancelled or uncertain new action must not leave an
            # earlier success eligible for a later shortcut.
            self._last_completed_head_target = None
            held = self._try_completed_head_hold(positions, completed)
            if held is not None:
                return held
            if (self._held_book_corners is not None
                    and not self._carried_head_transition_is_safe(*positions)):
                raise RuntimeError('Carried book or robot blocks requested head motion')
        if not client.wait_for_server(timeout_sec=min(5.0, self.timeout)):
            raise RuntimeError(f'action server unavailable for {list(names)}')
        with self._lock:
            payload_robot_contact = bool(
                getattr(self, '_payload_robot_watchdog_enabled', False)
                and getattr(self, '_target_robot_contact_latched', False)
            )
            if payload_robot_contact:
                self._payload_hazard_latched = 'payload_robot_contact'
        if payload_robot_contact:
            self._publish_status(
                'payload_hazard',
                reason='payload_robot_contact',
            )
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(names)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        command_duration = (initial_stow_owner.command_duration()
            if initial_stow_owner is not None else float(duration))
        if type(empty_pickup_setup) is not bool:
            raise ValueError('empty pickup setup timing marker must be bool')
        if empty_pickup_setup and (
                not getattr(self, 'empty_pickup_setup_retiming_enabled', False)
                or pre_send_check is None or trajectory_duration != duration):
            raise ValueError('empty setup timing requires its original goal and context check')
        if trajectory_duration is not None:
            if client is not self.arm_client or tuple(names) != tuple(ARM_JOINTS):
                raise ValueError('arm timing override requires the left arm controller and joints')
            command_duration = float(trajectory_duration)
            if (not math.isfinite(command_duration) or not math.isfinite(float(duration))
                    or not 0.0 < float(duration) / 3.0 <= command_duration <= float(duration)):
                raise ValueError('arm timing override must remain within the 1.0 to 3.0 speed range')
        # Keep duration unchanged for the original wall watchdog below.
        point.time_from_start = Duration(seconds=command_duration).to_msg()
        goal.trajectory.points = [point]
        if scene_reference is not None:
            _require_place_contact_clear(self)
            measured_scene_context(self, scene_reference)
        if self._cancel.is_set():
            return False
        if empty_torso_owner is not None:
            if trajectory_duration is not None or empty_pickup_setup or head_preflight:
                raise ValueError('empty torso owner cannot change another motion profile')
            with self._adaptive_command_guard():
                empty_torso_owner.check()
                if pre_send_check is not None:
                    pre_send_check()
                with self._lock:
                    require_follow_token_locked(self, _torso_hold_token, goal)
                    if getattr(self, '_empty_head_timing_owner', None) is not None:
                        raise ValueError('empty torso cannot dispatch while a head owner exists')
                    empty_torso_owner.require_send_locked(client, goal, _torso_hold_token)
                    acceptance = client.send_goal_async(goal)
                    empty_torso_owner.sent_locked(acceptance)
            goal_handle = empty_torso_owner.wait_acceptance(acceptance, self.timeout)
        elif empty_head_owner is not None:
            if not head_preflight or trajectory_duration is not None or empty_pickup_setup:
                raise ValueError('empty head owner requires the original head preflight')
            with self._adaptive_command_guard():
                _require_place_contact_clear(self)
                if pre_send_check is not None:
                    pre_send_check()
                with self._lock:
                    require_follow_token_locked(self, _torso_hold_token, goal)
                    empty_head_owner.admit_locked(client, goal)
                    acceptance = client.send_goal_async(goal)
                    empty_head_owner.sent_locked(acceptance)
            goal_handle = empty_head_owner.wait_acceptance(acceptance)
        elif initial_stow_owner is not None:
            with self._adaptive_command_guard():
                _require_place_contact_clear(self)
                with self._lock:
                    initial_stow_owner.admit_locked(client, goal, _torso_hold_token)
                    acceptance = client.send_goal_async(goal)
                    initial_stow_owner.sent_locked(acceptance)
            goal_handle = initial_stow_owner.wait_acceptance(acceptance, self.timeout)
        elif trajectory_duration is not None:
            velocity_record = None
            velocity_rejection = None
            try:
                with self._adaptive_command_guard():
                    _require_place_contact_clear(self)
                    if self._cancel.is_set():
                        return False
                    if pre_send_check is not None:
                        pre_send_check()
                    if self._cancel.is_set():
                        return False
                    with self._lock:
                        if self._cancel.is_set():
                            return False
                        try:
                            velocity_record = require_arm_velocity_locked(self, goal)
                            timing_factor = (2.0 if empty_pickup_setup else
                                getattr(self, 'additional_arm_time_scale', 1.0))
                            if timing_factor != 1.0:
                                goal, _, timing_record = retime_admitted_arm_goal(
                                    goal, velocity_record, timing_factor,
                                    nominal_duration=duration)
                                velocity_record = require_arm_velocity_locked(self, goal)
                                require_retimed_arm_headroom(velocity_record)
                                velocity_record['additional_arm_timing'] = timing_record
                                if self._cancel.is_set():
                                    raise ArmVelocityAdmissionRejected('cancelled before retimed arm dispatch', velocity_record)
                        except ArmVelocityAdmissionRejected as error:
                            velocity_rejection = error
                            raise
                        if _torso_hold_token is not None:
                            require_follow_token_locked(self, _torso_hold_token, goal)
                        acceptance = client.send_goal_async(goal)
            finally:
                if velocity_record is not None or velocity_rejection is not None:
                    publish_arm_velocity_admission(self,
                        velocity_record if velocity_rejection is None else velocity_rejection.record,
                        command='pick' if empty_pickup_setup else 'place',
                        admitted=velocity_rejection is None,
                        reason=None if velocity_rejection is None else str(velocity_rejection))
            goal_handle = self._wait_future(acceptance, self.timeout)
        elif (getattr(self, '_place_contact_guard', None) is not None
              or _torso_hold_token is not None):
            # The opt-in also serializes tracked sends against torso generations.
            # Do not hold the command lock while awaiting action acceptance.
            with self._adaptive_command_guard():
                _require_place_contact_clear(self)
                if self._cancel.is_set():
                    return False
                if pre_send_check is not None:
                    pre_send_check()
                if self._cancel.is_set():
                    return False
                if _torso_hold_token is None:
                    acceptance = client.send_goal_async(goal)
                else:
                    with self._lock:
                        require_follow_token_locked(self, _torso_hold_token, goal)
                        acceptance = client.send_goal_async(goal)
            goal_handle = self._wait_future(acceptance, self.timeout)
        else:
            if pre_send_check is not None:
                pre_send_check()
            if self._cancel.is_set():
                return False
            goal_handle = self._wait_future(client.send_goal_async(goal), self.timeout)
        if goal_handle is None or not goal_handle.accepted:
            return False
        with self._lock:
            self._goal_handles.append(goal_handle)
        keep_goal_registered = True
        try:
            result_future = goal_handle.get_result_async()
            if initial_stow_owner is not None:
                with self._lock:
                    initial_stow_owner.accepted_locked(goal_handle, result_future)
            if empty_torso_owner is not None:
                with self._lock:
                    empty_torso_owner.accepted_locked(goal_handle, result_future)
            if empty_head_owner is not None:
                with self._lock:
                    empty_head_owner.accepted_locked(goal_handle, result_future)
            wall_deadline = time.monotonic() + self.timeout + 4.0 * duration
            while not result_future.done():
                if initial_stow_owner is not None:
                    initial_stow_owner.check()
                if empty_torso_owner is not None:
                    empty_torso_owner.check()
                if empty_head_owner is not None:
                    empty_head_owner.check()
                if scene_reference is not None:
                    try:
                        _require_place_contact_clear(self)
                        measured_scene_context(self, scene_reference)
                    except RuntimeError:
                        if not self._cancel_goal_and_confirm(goal_handle, result_future):
                            keep_goal_registered = True
                            self._cancel.set()
                            raise RuntimeError('placement_scene_cancellation_unconfirmed')
                        keep_goal_registered = False
                        raise
                with self._lock:
                    payload_robot_contact = bool(
                        getattr(
                            self,
                            '_payload_robot_watchdog_enabled',
                            False,
                        )
                        and getattr(
                            self,
                            '_target_robot_contact_latched',
                            False,
                        )
                    )
                    if payload_robot_contact:
                        self._payload_hazard_latched = (
                            'payload_robot_contact'
                        )
                if payload_robot_contact:
                    self._publish_status(
                        'payload_hazard',
                        reason='payload_robot_contact',
                    )
                    if not self._cancel_goal_and_confirm(
                        goal_handle,
                        result_future,
                    ):
                        keep_goal_registered = True
                        raise RuntimeError(
                            'trajectory cancellation was not confirmed'
                        )
                    keep_goal_registered = False
                    return False
                if self._cancel.is_set():
                    if not self._cancel_goal_and_confirm(
                        goal_handle,
                        result_future,
                    ):
                        keep_goal_registered = True
                        raise RuntimeError(
                            'trajectory cancellation was not confirmed'
                        )
                    keep_goal_registered = False
                    return False
                if time.monotonic() >= wall_deadline:
                    if not self._cancel_goal_and_confirm(
                        goal_handle,
                        result_future,
                    ):
                        keep_goal_registered = True
                        self._cancel.set()
                        raise RuntimeError(
                            'trajectory cancellation was not confirmed'
                        )
                    keep_goal_registered = False
                    raise TimeoutError('trajectory timed out')
                time.sleep(0.02)
            if initial_stow_owner is not None:
                initial_stow_owner.terminal(result_future)
            if empty_torso_owner is not None:
                empty_torso_owner.action_completed(result_future)
            if empty_head_owner is not None:
                empty_head_owner.terminal(result_future)
            # The result future and a contact callback can become ready in the
            # same executor turn.  Check the latched collision once more after
            # leaving the polling loop so a nominal controller success cannot
            # win that race.
            with self._lock:
                payload_robot_contact = bool(
                    getattr(self, '_payload_robot_watchdog_enabled', False)
                    and getattr(self, '_target_robot_contact_latched', False)
                )
                if payload_robot_contact:
                    self._payload_hazard_latched = 'payload_robot_contact'
            if payload_robot_contact:
                self._publish_status(
                    'payload_hazard',
                    reason='payload_robot_contact',
                )
                keep_goal_registered = False
                return False
            keep_goal_registered = False
            if self._cancel.is_set():
                return False
            if scene_reference is not None:
                _require_place_contact_clear(self)
                measured_scene_context(self, scene_reference)
            wrapped = result_future.result()
            keep_goal_registered = False
            succeeded = wrapped.status == GoalStatus.STATUS_SUCCEEDED
            if head_preflight and succeeded and not self._cancel.is_set():
                with self._lock:
                    self._last_completed_head_target = (
                        tuple(float(value) for value in positions),
                        int(self.get_clock().now().nanoseconds),
                    )
            return succeeded
        finally:
            if not keep_goal_registered:
                with self._lock:
                    if goal_handle in self._goal_handles:
                        self._goal_handles.remove(goal_handle)

    def _head_hold_snapshot(self, target, completed):
        """Return fresh stationary measured geometry, never a cached verdict."""
        if completed is None or tuple(float(v) for v in target) != completed[0]:
            return None
        with self._lock:
            return self._head_hold_snapshot_locked(target, completed)

    def _head_hold_snapshot_locked(self, target, completed, *, require_target=True):
        """Caller owns the non-reentrant sensor lock; acquire no lock here."""
        names = (*IK_JOINTS, *RIGHT_ARM_JOINTS, *HEAD_JOINTS)
        joints = dict(self.joints)
        velocities = dict(getattr(self, '_joint_velocities', {}))
        stamps = dict(getattr(self, '_joint_stamps_ns', {}))
        held = getattr(self, '_held_book_corners', None)
        attached = None if held is None else np.asarray(held, dtype=float).copy()
        # Only the retained-book look_bin case opts into this shortcut.
        # Empty-head callers keep their prior action/preflight behavior.
        if attached is None:
            return None
        now = int(self.get_clock().now().nanoseconds)
        if now < completed[1]:
            return None
        for name in names:
            value, velocity = joints.get(name, math.nan), velocities.get(name, math.nan)
            # These strict bounds select a no-command opportunity only. They
            # do not replace any commanded-motion or collision tolerance.
            maximum_speed = 1e-5 if name == 'torso_lift_joint' else 1e-4
            if (not math.isfinite(value) or not math.isfinite(velocity)
                    or abs(velocity) > maximum_speed
                    or not -.05e9 <= now - stamps.get(name, -10**18) <= .15e9):
                return None
        if any(stamps[n] <= completed[1] for n in HEAD_JOINTS):
            return None
        if require_target and any(abs(joints[n] - float(v)) > 1e-6 for n, v in zip(HEAD_JOINTS, target)):
            return None
        if attached is not None and (attached.shape != (8, 3) or not np.all(np.isfinite(attached))):
            return None
        return dict(joints={n: float(joints[n]) for n in names},
                    velocities={n: float(velocities[n]) for n in names},
                    producer_stamps_ns={n: int(stamps[n]) for n in names},
                    evaluated_ros_ns=now, attached=attached,
                    contact_generation=int(getattr(self, '_contact_generation', 0)),
                    contact_epoch=int(getattr(self, '_contact_epoch', 0)),
                    feedback_identity=(getattr(self, '_gripper_feedback_samples', ())[-1]
                                       if getattr(self, '_gripper_feedback_samples', ()) else None))

    def _head_hold_admission_clear(self):
        """Short gates under command ownership, never inside the sensor lock.

        Sensor helpers below acquire the node's non-reentrant lock themselves.
        """
        if (self._cancel.is_set() or getattr(self, '_goal_handles', ())
                or getattr(self, '_pending_retained_acceptances', ())):
            return False
        _require_place_contact_clear(self)
        reference = getattr(self, '_active_place_scene_reference', None)
        if reference is not None:
            measured_scene_context(self, reference)
        if getattr(self, '_held_book_corners', None) is not None:
            reason = self._payload_hazard_reason(max_age=.15)
            if reason is not None:
                self._publish_status('payload_hazard', reason=reason)
                return False
        if (getattr(self, '_payload_robot_watchdog_enabled', False)
                and getattr(self, '_target_robot_contact_latched', False)):
            self._payload_hazard_latched = 'payload_robot_contact'
            return False
        return True

    def _head_hold_geometry_is_safe(self, snapshot):
        """Check the current body and one measured head pose using one context."""
        joints = snapshot['joints']
        q = np.asarray([joints[n] for n in IK_JOINTS], dtype=float)
        right = np.asarray([joints[n] for n in RIGHT_ARM_JOINTS], dtype=float)
        head = np.asarray([joints[n] for n in HEAD_JOINTS], dtype=float)
        if (np.any(head < self.head_chain.lower[1:])
                or np.any(head > self.head_chain.upper[1:])):
            return False
        context = dict(right_positions=right, head_positions=head)
        if self._robot_self_collision(q, **context) is not None:
            return False
        attached = snapshot['attached']
        if attached is None:
            return True
        hand = self.chain.forward(q)
        corners = attached @ hand[:3, :3].T + hand[:3, 3]
        return (self._carried_robot_collision(q, corners, **context) is None
                and self._head_motion_collision(q, head, corners,
                                                right_positions=right) is None)

    def _try_completed_head_hold(self, target, completed):
        """Reuse a checked hold, or reacquire before the ordinary checked action."""
        if completed is None or tuple(float(v) for v in target) != completed[0]:
            return None
        with self._lock:
            held = getattr(self, '_held_book_corners', None)
            if held is None:
                return None
            identity = dict(contact_epoch=int(getattr(self, '_contact_epoch', 0)),
                            attached=np.asarray(held, dtype=float).copy())

        def report(event, reason, **fields):
            # Diagnostics must not change admission or execute under the sensor lock.
            try:
                self._publish_status(event, reason=reason, target=list(target), **fields)
            except Exception:
                pass

        def hard_fault_locked():
            # Called only with the plain sensor lock; acquire no other lock here.
            if self._cancel.is_set():
                return 'cancelled'
            if getattr(self, '_goal_handles', ()):
                return 'active_goal'
            if getattr(self, '_pending_retained_acceptances', ()):
                return 'pending_goal_acceptance'
            for name in ('_payload_hazard_latched', '_held_grip_sensor_fault'):
                if getattr(self, name, None) is not None:
                    return str(getattr(self, name))
            if getattr(self, '_target_robot_contact_latched', False):
                return 'payload_robot_contact'
            if int(getattr(self, '_contact_epoch', 0)) != identity['contact_epoch']:
                return 'contact_identity_changed'
            current = getattr(self, '_held_book_corners', None)
            attached = identity['attached']
            if (current is None or attached.shape != (8, 3)
                    or not np.all(np.isfinite(attached))
                    or not np.array_equal(attached, current)):
                return 'attachment_changed_or_invalid'
            if int(self.get_clock().now().nanoseconds) < completed[1]:
                return 'clock_reversed'
            return None

        def fallback(reason, *, recheck=False, deadline=None):
            # A failed shortcut is not permission to feed stale joints to the old
            # sweep. Reacquire its fresh stationary start, with a short wall bound.
            report('head_hold_fallback', reason, phase='waiting_for_fresh_stationary_start')
            if deadline is None:
                deadline = time.monotonic() + min(2.0, max(0.0, float(self.timeout)))
            attempts = 0
            while True:
                attempts += 1
                fault = None
                with self._adaptive_command_guard():
                    if not self._head_hold_admission_clear():
                        fault = 'admission_interlock'
                    with self._lock:
                        fault = hard_fault_locked() or fault
                        fresh = (None if fault else self._head_hold_snapshot_locked(
                            target, completed, require_target=False))
                if fault:
                    report('head_hold_rejected', fault, fallback_reason=reason, attempts=attempts)
                    return False
                if fresh is not None:
                    report('head_hold_fallback', reason, phase='fresh_start_ready', attempts=attempts,
                           producer_stamps_ns=fresh['producer_stamps_ns'],
                           measured_joint_velocities=fresh['velocities'])
                    # A fresh start alone does not establish the exact target.
                    # Only the initial snapshot miss may request this one retry;
                    # the original geometry/final-lock proof below still runs.
                    if recheck and time.monotonic() < deadline:
                        strict = self._head_hold_snapshot(target, completed)
                        if strict is not None:
                            report('head_hold_fallback', reason,
                                   phase='strict_target_recheck_ready', attempts=attempts)
                            return strict, deadline
                    # None retains the complete sweep/server/controller path.
                    return None
                if time.monotonic() >= deadline:
                    report('head_hold_rejected', 'fresh_stationary_start_timeout',
                           fallback_reason=reason, attempts=attempts)
                    return False
                time.sleep(.02)

        recheck_deadline = None
        snapshot = self._head_hold_snapshot(target, completed)
        if snapshot is None:
            recovered = fallback('initial_snapshot_ineligible', recheck=(
                getattr(self, 'completed_head_hold_recheck_enabled', False) is True))
            if not isinstance(recovered, tuple):
                return recovered
            snapshot, recheck_deadline = recovered
        with self._adaptive_command_guard():
            admitted = self._head_hold_admission_clear()
            with self._lock:
                fault = hard_fault_locked()
        if fault or not admitted:
            report('head_hold_rejected', fault or 'admission_interlock')
            return False
        # Reuse the first waiting window; never start a second retry window.
        if recheck_deadline is not None and time.monotonic() >= recheck_deadline:
            return fallback('recheck_window_expired', deadline=recheck_deadline)
        # No command or sensor lock spans geometry, callbacks or action waits.
        if not self._head_hold_geometry_is_safe(snapshot):
            report('head_hold_rejected', 'current_geometry_unsafe')
            raise RuntimeError('Current head hold geometry is unsafe')
        reason = None
        fields = None
        with self._adaptive_command_guard():
            checked = self._head_hold_snapshot(target, completed)
            admitted = self._head_hold_admission_clear()
            with self._lock:
                fault = hard_fault_locked()
                latest = self._head_hold_snapshot_locked(target, completed)
                if not fault and not admitted:
                    fault = 'admission_interlock'
                if not fault:
                    if checked is None:
                        reason = 'checked_snapshot_ineligible'
                    elif latest is None:
                        reason = 'latest_snapshot_ineligible'
                    elif any(abs(latest['joints'][n] - value) > 1e-6
                             for n, value in snapshot['joints'].items()):
                        reason = 'geometry_snapshot_joint_changed'
                    elif (latest['contact_generation'] != checked['contact_generation']
                            or latest['feedback_identity'] is not checked['feedback_identity']):
                        reason = 'sensor_delivery_changed'
                        if latest['feedback_identity'] is not checked['feedback_identity']:
                            feedback = latest['feedback_identity']
                            if (feedback is None or not all(math.isfinite(getattr(feedback, name, math.nan))
                                    for name in ('position', 'velocity', 'effort'))
                                    or not -100_000_000 <= latest['evaluated_ros_ns'] - getattr(
                                        feedback, 'stamp_ns', -10**18) <= 150_000_000):
                                fault = 'invalid_retained_feedback'
                    if (not fault and reason is None and recheck_deadline is not None
                            and time.monotonic() >= recheck_deadline):
                        reason = 'recheck_window_expired'
                    if not fault and reason is None:
                        self._last_completed_head_target = completed
                        fields = dict(reason='completed_target_measured_stationary',
                            target=list(completed[0]), completed_action_ros_ns=completed[1],
                            measured_head=[latest['joints'][n] for n in HEAD_JOINTS],
                            measured_joint_velocities=latest['velocities'],
                            producer_stamps_ns=latest['producer_stamps_ns'],
                            maximum_joint_change=max(abs(latest['joints'][n]-v)
                                                     for n, v in snapshot['joints'].items()),
                            collision_scope='current measured body and nominal held book/head; no commanded sweep')
            if fields is not None:
                self._publish_status('head_motion_skipped', **fields)
                return not self._cancel.is_set()
        if fault:
            report('head_hold_rejected', fault)
            return False
        return fallback(reason, deadline=recheck_deadline)

    def _move_head(self, pan: float, tilt: float, *, empty_head_timing=None) -> bool:
        if empty_head_timing is not None:
            return _empty_head_timing.run(self, empty_head_timing, pan, tilt)
        return self._follow(
            self.head_client,
            HEAD_JOINTS,
            [pan, tilt],
            1.2,
            head_preflight=True,
        )

    def _wait_sim_duration(self, duration: float) -> bool:
        if (getattr(self, '_adaptive_close_active', False)
                and getattr(self, '_adaptive_motion_started', False)):
            return self._wait_adaptive_gripper_duration(duration)
        started_ns = self.get_clock().now().nanoseconds
        wall_deadline = time.monotonic() + max(self.timeout, duration * 4.0)
        required_ns = int(max(0.0, duration) * 1e9)
        while not self._cancel.is_set() and time.monotonic() < wall_deadline:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns >= started_ns and now_ns - started_ns >= required_ns:
                return True
            time.sleep(0.02)
        return False

    def _publish_gripper_position(
        self,
        position: float,
        *,
        motion_seconds: float,
        wait_seconds: float,
        respect_cancel: bool = False,
    ) -> bool:
        """Publish a position command; openings opt into cancellation admission."""
        message = JointTrajectory()
        message.joint_names = ['gripper_left_finger_joint']
        point = JointTrajectoryPoint()
        point.positions = [float(np.clip(position, 0.0, 0.069))]
        point.time_from_start = Duration(seconds=motion_seconds).to_msg()
        message.points = [point]
        for _ in range(2):
            if respect_cancel:
                # Recheck every duplicate under the same lock as a safety hold.
                # A hold after the first publish must not be replaced by a
                # delayed second opening. Never keep this lock during a wait.
                with self._adaptive_command_guard():
                    cancel = getattr(self, '_cancel', None)
                    if _place_contact_fault(self) or (cancel is not None and cancel.is_set()):
                        return False
                    self.gripper_pub.publish(message)
            else:
                self.gripper_pub.publish(message)
            time.sleep(0.04)
        return self._wait_sim_duration(wait_seconds)

    def _command_gripper(
        self, position: float, *, respect_cancel: bool = False,
    ) -> bool:
        return self._publish_gripper_position(
            position,
            motion_seconds=0.8,
            wait_seconds=self.gripper_settle,
            respect_cancel=respect_cancel,
        )

    def _command_adaptive_gripper_step(self, position: float) -> bool:
        return self._publish_adaptive_gripper_position(
            position,
            motion_seconds=self.adaptive_step_motion,
            wait_seconds=(
                self.adaptive_step_motion + self.adaptive_step_settle
            ),
        )

    def _adaptive_command_guard(self):
        lock = getattr(self, '_adaptive_command_lock', None)
        if lock is None:
            lock = threading.RLock()
            self._adaptive_command_lock = lock
        return lock

    def _adaptive_motion_feedback(self):
        """Copy a usable measurement; never pretend an old value is current."""
        feedback = self._latest_gripper_feedback()
        if feedback is None:
            return None, 'joint_feedback_unavailable'
        now_ns = int(self.get_clock().now().nanoseconds)
        if (not math.isfinite(feedback.position)
                or not math.isfinite(feedback.velocity)
                or not (_stock_measured_position_in_range(feedback.position)
                        if _stock_diagnostic(self) else 0.0 <= feedback.position <= 0.069)):
            return None, 'invalid_gripper_motion_feedback'
        if not -100_000_000 <= now_ns - feedback.stamp_ns <= 150_000_000:
            return None, 'stale_gripper_motion_feedback'
        return feedback, None

    def _hold_adaptive_gripper(self, reason: str) -> None:
        """Serialize a one-shot measured-position hold without cancelling recovery."""
        with self._adaptive_command_guard():
            if ((getattr(self, '_release_pose_owner', None) is not None
                    or getattr(self, '_release_pose_fault_latched', None) is not None)
                    and release_pose_finish.block_postopen_gripper_hold(self)):
                return
            if getattr(self, '_adaptive_motion_halt_reason', None) is None:
                self._adaptive_motion_halt_reason = str(reason)
            if getattr(self, '_adaptive_hold_sent', False):
                return
            feedback, feedback_error = self._adaptive_motion_feedback()
            # Mark the hold attempt even when feedback is unusable. In that
            # case publish no invented position and return promptly to the
            # checked reopen/recovery path; no further inward step is allowed.
            self._adaptive_hold_sent = True
            published = False
            hold_position = None
            if feedback is not None:
                message = JointTrajectory()
                message.joint_names = ['gripper_left_finger_joint']
                point = JointTrajectoryPoint()
                # Only an already-valid stock endpoint roundoff reading may
                # be normalized for publication; retain the measured value.
                hold_position = float(feedback.position)
                if _stock_diagnostic(self):
                    hold_position = min(.069, max(0.0, hold_position))
                point.positions = [hold_position]
                point.velocities = [0.0]
                point.time_from_start = Duration(seconds=0.02).to_msg()
                message.points = [point]
                self.gripper_pub.publish(message)
                published = True
            self._publish_status(
                'adaptive_gripper_interrupted', reason=str(reason),
                hold_published=published,
                commanded_hold_position=hold_position,
                measured_position=(float(feedback.position) if feedback is not None else None),
                feedback_reason=feedback_error,
            )

    def _interrupt_adaptive_gripper_if_fault(self) -> None:
        # Called only after a sensor callback releases _lock. The sole lock
        # ordering is command lock -> short sensor snapshot -> publication;
        # no callback can hold the sensor lock while waiting for a command.
        if not getattr(self, '_adaptive_motion_started', False):
            return
        with self._adaptive_command_guard():
            if (not getattr(self, '_adaptive_close_active', False)
                    or not getattr(self, '_adaptive_motion_started', False)):
                return
            reason = self._adaptive_overload_reason()
            if reason is not None:
                self._hold_adaptive_gripper(reason)

    def _publish_adaptive_gripper_position(
        self, position: float, *, motion_seconds: float, wait_seconds: float,
    ) -> bool:
        """One publish, with no delayed duplicate that could replace a hold."""
        with self._adaptive_command_guard():
            reason = (getattr(self, '_adaptive_motion_halt_reason', None)
                      or self._adaptive_overload_reason())
            feedback, feedback_error = self._adaptive_motion_feedback()
            if reason or feedback_error:
                self._hold_adaptive_gripper(reason or feedback_error)
                return False
            if (not math.isfinite(position) or not 0.0 <= position <= 0.069
                    or not math.isfinite(motion_seconds) or motion_seconds <= 0.0
                    or not math.isfinite(wait_seconds) or wait_seconds < motion_seconds):
                self._hold_adaptive_gripper('invalid_adaptive_position_command')
                return False
            message = JointTrajectory()
            message.joint_names = ['gripper_left_finger_joint']
            point = JointTrajectoryPoint()
            point.positions = [float(position)]
            point.time_from_start = Duration(seconds=motion_seconds).to_msg()
            message.points = [point]
            controller = getattr(self, '_stock_close_controller', None)
            if controller is not None:
                controller.before_publish(position, motion_seconds, wait_seconds)
                # A new stock close invalidates the earlier measured-open proof.
                # This is not closed/retained evidence, including if publish fails.
                lock = getattr(self, '_lock', None)
                if lock is None:
                    self._gripper_open_confirmed = False
                else:
                    with lock:
                        self._gripper_open_confirmed = False
            self._adaptive_motion_started = True
            self.gripper_pub.publish(message)
        return self._wait_adaptive_gripper_duration(wait_seconds)

    def _wait_adaptive_gripper_duration(self, duration: float) -> bool:
        controller = getattr(self, '_stock_close_controller', None)
        if controller is not None: return controller.wait(duration)
        started_ns = int(self.get_clock().now().nanoseconds)
        wall_deadline = time.monotonic() + max(self.timeout, duration * 4.0)
        required_ns = int(max(0.0, duration) * 1e9)
        while time.monotonic() < wall_deadline:
            reason = (getattr(self, '_adaptive_motion_halt_reason', None)
                      or self._adaptive_overload_reason())
            _, feedback_error = self._adaptive_motion_feedback()
            if self._cancel.is_set():
                reason = reason or 'cancelled'
            now_ns = int(self.get_clock().now().nanoseconds)
            if now_ns < started_ns:
                reason = reason or 'clock_reversed'
            if reason or feedback_error:
                self._hold_adaptive_gripper(reason or feedback_error)
                return False
            if now_ns - started_ns >= required_ns:
                return True
            time.sleep(0.002)
        self._hold_adaptive_gripper('adaptive_gripper_motion_timeout')
        return False

    def _open_gripper(self, *, verify_measurement=None) -> bool:
        """Open the left gripper and clear retained-payload state on success."""
        cancel = getattr(self, '_cancel', None)
        if _place_contact_fault(self) or (cancel is not None and cancel.is_set()):
            return False
        scene_reference = getattr(self, '_active_place_scene_reference', None)
        if scene_reference is not None:
            measured_scene_context(self, scene_reference)
        if getattr(self, 'preclose_aperture_geometry_enabled', False):
            from .preclose_aperture_geometry import invalidate
            invalidate(self)
        lock = getattr(self, '_lock', None)
        if lock is None:
            self._gripper_open_confirmed = False
            monitor_was_enabled = bool(
                getattr(self, '_payload_monitor_enabled', False)
            )
            self._payload_monitor_enabled = False
            self._retention_probe_active = True
        else:
            with lock:
                self._gripper_open_confirmed = False
                monitor_was_enabled = bool(
                    getattr(self, '_payload_monitor_enabled', False)
                )
                self._payload_monitor_enabled = False
                self._retention_probe_active = True

        def restore_monitor_after_failed_open() -> None:
            if lock is None:
                self._payload_monitor_enabled = monitor_was_enabled
                self._retention_probe_active = False
            else:
                with lock:
                    self._payload_monitor_enabled = monitor_was_enabled
                    self._retention_probe_active = False

        try:
            opened = self._command_gripper(self.gripper_open, respect_cancel=True)
        except Exception:
            restore_monitor_after_failed_open()
            raise
        if not opened:
            restore_monitor_after_failed_open()
            return False
        if verify_measurement is not None:
            try:
                measurement_ok = bool(verify_measurement())
            except Exception:
                restore_monitor_after_failed_open()
                raise
            if not measurement_ok:
                restore_monitor_after_failed_open()
                return False

        if _place_contact_fault(self):
            restore_monitor_after_failed_open()
            return False

        def clear_released_payload() -> None:
            self._held_book_corners = None
            self._carried_staging_solution = None
            self._cached_post_retreat_plan = None
            self._post_retreat_shelf_front_x = None
            self._gravity_supported_payload = False
            self._supported_post_retreat_staging_required = False
            self._transport_lock_engaged = False
            self._gripper_open_confirmed = True
            self._held_grip_sensor_fault = None
            self._payload_hazard_latched = None
            self._payload_monitor_enabled = False
            self._retention_probe_active = False
            self._payload_robot_watchdog_enabled = False
            self._clear_target_contact_samples_unlocked(
                reset_robot_contact=True,
                reset_target_model=True,
            )

        if lock is None:
            clear_released_payload()
        else:
            with lock:
                clear_released_payload()
        return True

    def _contact_timestamp_is_recent(
        self,
        timestamp_ns: int,
        *,
        max_age: Optional[float] = None,
    ) -> bool:
        if timestamp_ns <= 0:
            return False
        age = (self.get_clock().now().nanoseconds - timestamp_ns) / 1e9
        permitted_age = (
            self.grasp_contact_max_age if max_age is None else float(max_age)
        )
        return -0.10 <= age <= permitted_age

    def _clear_target_contact_samples(
        self,
        *,
        reset_robot_contact: bool = False,
        reset_target_model: bool = False,
    ) -> None:
        lock = getattr(self, '_lock', None)
        if lock is None:
            self._clear_target_contact_samples_unlocked(
                reset_robot_contact=reset_robot_contact,
                reset_target_model=reset_target_model,
            )
            return
        with lock:
            self._clear_target_contact_samples_unlocked(
                reset_robot_contact=reset_robot_contact,
                reset_target_model=reset_target_model,
            )

    def _clear_target_contact_samples_unlocked(
        self,
        *,
        reset_robot_contact: bool = False,
        reset_target_model: bool = False,
    ) -> None:
        """Clear contact state while the caller owns ``_lock``, when present."""
        self._left_target_contact_ns = 0
        self._right_target_contact_ns = 0
        self._book_contact_samples = {}
        self._book_contact_force_samples = {}
        self._book_contact_force_frames = {}
        self._contact_generation = int(
            getattr(self, '_contact_generation', 0)
        ) + 1
        self._contact_epoch = int(getattr(self, '_contact_epoch', 0)) + 1
        if reset_robot_contact:
            self._target_robot_contact_latched = False
        if reset_target_model:
            self._target_book_model = None

    def _target_contact_sides(
        self,
        *,
        max_age: Optional[float] = None,
    ) -> Tuple[bool, bool]:
        lock = getattr(self, '_lock', None)
        if lock is None:
            left_timestamp = self._left_target_contact_ns
            right_timestamp = self._right_target_contact_ns
        else:
            with lock:
                left_timestamp = self._left_target_contact_ns
                right_timestamp = self._right_target_contact_ns
        return (
            self._contact_timestamp_is_recent(
                left_timestamp,
                max_age=max_age,
            ),
            self._contact_timestamp_is_recent(
                right_timestamp,
                max_age=max_age,
            ),
        )

    def _target_contact_recent(self, *, max_age: Optional[float] = None) -> bool:
        left_contact, right_contact = self._target_contact_sides(max_age=max_age)
        if getattr(self, '_gravity_supported_payload', False):
            # The negative q7 cradle keeps local +y pointing upward, so the
            # physical left finger (local -y) is the lower load-bearing jaw.
            # Once the book is cradled, gravity can legitimately separate it
            # from the upper/right jaw; fresh left contact is the retention
            # signal.  Before that transition a true pinch still requires both
            # sides.
            return left_contact
        return left_contact and right_contact

    def _pinch_sample(
        self,
        *,
        max_age: Optional[float] = None,
    ) -> Tuple[bool, float, bool, bool, bool]:
        """Measure the current contact-mode and jaw-width retention gate."""
        lock = getattr(self, '_lock', None)
        if lock is None:
            grasp_width = float(
                self.joints.get('gripper_left_finger_joint', 0.0)
            )
        else:
            with lock:
                grasp_width = float(
                    self.joints.get('gripper_left_finger_joint', 0.0)
                )
        if max_age is None:
            left_contact, right_contact = self._target_contact_sides()
        else:
            left_contact, right_contact = self._target_contact_sides(
                max_age=max_age
            )
        if _stock_diagnostic(self):
            # Actuator coordinate, not a measured pad gap or empty-grasp proof.
            plausible_width = _stock_measured_position_in_range(grasp_width)
        elif getattr(self, '_transport_lock_engaged', False):
            # The lock is reachable only after a normal-width bilateral pinch
            # has already been proved without opening. Adaptive closure uses
            # a bounded preload relative to measured first contact; the lock
            # position is its travel floor, not the physical book thickness.
            # Excess closure can extrude the book. Contact remains
            # mandatory, so the bounded servo tolerance cannot turn an empty
            # gripper into a retained payload.
            plausible_width = (
                self.gripper_transport_lock - 0.001
                <= grasp_width
                < self.grasp_max_position
            )
        else:
            plausible_width = (
                self.grasp_min_position + self.grasp_min_margin
                <= grasp_width
                < self.grasp_max_position
            )
        retained_contact = (
            left_contact
            if getattr(self, '_gravity_supported_payload', False)
            else left_contact and right_contact
        )
        verified = retained_contact and plausible_width
        return verified, grasp_width, left_contact, right_contact, plausible_width

    def _latest_gripper_feedback(self) -> Optional[GripperFeedback]:
        lock = getattr(self, '_lock', None)
        if lock is None:
            samples = tuple(
                getattr(self, '_gripper_feedback_samples', ())
            )
        else:
            with lock:
                samples = tuple(
                    getattr(self, '_gripper_feedback_samples', ())
                )
        return samples[-1] if samples else None

    def _adaptive_effort_baseline(self, now_ns: int) -> float:
        """Return a recent robust effort baseline, or NaN when unavailable."""
        lock = getattr(self, '_lock', None)
        if lock is None:
            samples = tuple(
                getattr(self, '_gripper_feedback_samples', ())
            )
        else:
            with lock:
                samples = tuple(
                    getattr(self, '_gripper_feedback_samples', ())
                )
        efforts = [
            float(sample.effort)
            for sample in samples[-15:]
            if math.isfinite(float(sample.effort))
            and -100_000_000
            <= int(now_ns) - int(sample.stamp_ns)
            <= 300_000_000
        ]
        return float(np.median(efforts)) if efforts else math.nan

    def _adaptive_force_histories(
        self,
    ) -> Tuple[
        Optional[str],
        Tuple[ForceSample, ...],
        Tuple[ForceSample, ...],
        Optional[str],
    ]:
        """Copy force histories and report ambiguous/anonymous identity."""
        lock = getattr(self, '_lock', None)

        def snapshot():
            selected_model = getattr(self, '_target_book_model', None)
            source = getattr(self, '_book_contact_force_samples', {})
            copied = {
                model: (tuple(histories[0]), tuple(histories[1]))
                for model, histories in source.items()
            }
            return selected_model, copied

        if lock is None:
            selected_model, samples = snapshot()
        else:
            with lock:
                selected_model, samples = snapshot()
        if selected_model is not None:
            left, right = samples.get(selected_model, ((), ()))
            return selected_model, left, right, None
        concrete = [
            (model, histories)
            for model, histories in samples.items()
            if model and (histories[0] or histories[1])
        ]
        if len(concrete) > 1:
            return None, (), (), 'ambiguous_target_contact'
        if len(concrete) == 1:
            model, (left, right) = concrete[0]
            return model, left, right, 'target_not_latched'
        anonymous = samples.get('', ((), ()))
        if anonymous[0] or anonymous[1]:
            return None, anonymous[0], anonymous[1], 'anonymous_target_contact'
        return None, (), (), None

    def _adaptive_pressure_evidence(
        self,
        *,
        minimum_width: float,
        baseline_effort: float,
    ) -> Tuple[AdaptiveGraspEvidence, Optional[str], Optional[str]]:
        model, left, right, identity_reason = (
            self._adaptive_force_histories()
        )

        feedback = self._latest_gripper_feedback()
        # Copy every input before reading the evaluation clock. A newer joint
        # callback after the wait must not replace this fixed snapshot and then
        # look falsely future-dated against the clock we already captured.
        now_ns = int(self.get_clock().now().nanoseconds)
        latest_stamp = max(
            (sample.stamp_ns for sample in (*left, *right)), default=now_ns,
        )
        feedback_stamp = getattr(feedback, 'stamp_ns', None)
        if (isinstance(feedback_stamp, (int, np.integer))
                and not isinstance(feedback_stamp, (bool, np.bool_))):
            latest_stamp = max(latest_stamp, int(feedback_stamp))
        if now_ns < latest_stamp <= now_ns + 100_000_000:
            # Wait for /clock to catch this fixed snapshot, rather than shifting
            # sensor stamps or continually chasing the next 500 Hz measurement.
            # No further inward command can run while chronology is pending.
            deadline = time.monotonic() + 0.5
            while now_ns < latest_stamp and time.monotonic() < deadline:
                cancel = getattr(self, '_cancel', None)
                if (cancel is not None and cancel.is_set()
                        or self._adaptive_overload_reason() is not None):
                    break
                time.sleep(0.002)
                now_ns = int(self.get_clock().now().nanoseconds)
        if _stock_diagnostic(self):
            evidence = _stock_contact_evidence(left, right, feedback, now_ns,
                maximum_velocity=self.adaptive_velocity_tolerance)
            overload = self._adaptive_overload_reason()
            if overload: evidence = replace(evidence, verified=False, reason=overload)
            return evidence, model, identity_reason
        evidence = evaluate_bilateral_contact(
            left,
            right,
            feedback,
            now_ns,
            minimum_width=minimum_width,
            maximum_width=self.grasp_max_position,
            minimum_force=self.adaptive_contact_force_minimum,
            maximum_force=self.adaptive_contact_force_maximum,
            minimum_samples=self.adaptive_contact_samples,
            maximum_age_seconds=self.adaptive_contact_max_age,
            maximum_gap_seconds=self.adaptive_contact_max_gap,
            minimum_span_seconds=self.adaptive_contact_min_span,
            maximum_side_skew_seconds=self.adaptive_contact_max_skew,
            maximum_velocity=self.adaptive_velocity_tolerance,
            maximum_effort=self.adaptive_effort_maximum,
            baseline_effort=baseline_effort,
            maximum_effort_delta=self.adaptive_effort_delta_maximum,
        )
        overload = self._adaptive_overload_reason()
        if overload is not None:
            evidence = replace(evidence, verified=False, reason=overload)
        return evidence, model, identity_reason

    def _publish_adaptive_close_result(
        self,
        evidence: AdaptiveGraspEvidence,
        *,
        verified: bool,
        reason: str,
        close_stage: str,
        commanded_position: float,
        acquisition_position: Optional[float],
    ) -> Tuple[bool, float, bool, bool, bool]:
        self._transport_lock_engaged = bool(verified)
        sample = self._pinch_sample(max_age=self.adaptive_contact_max_age)
        verified = bool(verified and sample[0])
        if not verified:
            self._transport_lock_engaged = False
        result = (verified, *sample[1:])
        self._publish_status(
            'gripper_closed',
            control_mode=('stock_public_position' if _stock_diagnostic(self) else 'adaptive_pressure'),
            pressure_qualified=bool(verified and not _stock_diagnostic(self)),
            stock_effort_peak=getattr(self, '_stock_effort_peak', None),
            close_stage=close_stage,
            reason=reason,
            commanded_position=commanded_position,
            acquisition_position=acquisition_position,
            acquisition_plausible_width=(
                acquisition_position is not None
                and (_stock_measured_position_in_range(acquisition_position) if _stock_diagnostic(self) else
                     self.grasp_min_position+self.grasp_min_margin <= acquisition_position < self.grasp_max_position)
            ),
            transport_lock_engaged=bool(
                getattr(self, '_transport_lock_engaged', False)
            ),
            measured_position=sample[1],
            left_contact=sample[2],
            right_contact=sample[3],
            bilateral_contact=sample[2] and sample[3],
            plausible_width=sample[4],
            left_force_newtons=(
                evidence.left_force
                if math.isfinite(evidence.left_force)
                else None
            ),
            right_force_newtons=(
                evidence.right_force
                if math.isfinite(evidence.right_force)
                else None
            ),
            joint_effort=(
                evidence.effort if math.isfinite(evidence.effort) else None
            ),
            joint_effort_delta=(
                evidence.effort_delta
                if math.isfinite(evidence.effort_delta)
                else None
            ),
            left_force_samples=evidence.left_samples,
            right_force_samples=evidence.right_samples,
            force_peaks=dict(getattr(self, '_adaptive_force_peaks', {})),
        )
        return result

    @staticmethod
    def _adaptive_evidence_is_terminal(reason: str) -> bool:
        """Return whether another inward step would be unsafe or uninformed."""
        if 'overload' in reason:
            return True
        if reason in {
            'implausible_gripper_width',
            'missing_gripper_feedback',
            'stale_gripper_feedback',
            'joint_feedback_invalid',
            'joint_feedback_unavailable',
            'joint_feedback_stale',
            'force_history_invalid',
            'width_too_narrow',
            'joint_still_moving',
        }:
            return True
        return reason.startswith(
            (
                'invalid_',
                'malformed_',
                'nonfinite_',
                'negative_',
                'nonmonotonic_',
                'future_',
            )
        )

    def _adaptive_overload_reason(self) -> Optional[str]:
        lock = getattr(self, '_lock', None)
        if lock is None:
            return getattr(self, '_adaptive_overload_latched', None)
        with lock:
            return getattr(self, '_adaptive_overload_latched', None)

    def _record_adaptive_force_peak(
        self, model: str, side: int, force: float, stamp_ns: int,
    ) -> None:
        """Retain measured peak diagnostics independently of confirmation windows.

        Contact updates call this while holding the existing contact lock.
        These diagnostics never authorize closure or weaken an overload gate.
        """
        if not (getattr(self, '_adaptive_close_active', False) or (
                _stock_diagnostic(self) and getattr(self, '_held_book_corners', None) is not None)):
            return
        if not math.isfinite(force):
            return
        peaks = getattr(self, '_adaptive_force_peaks', {})
        key = 'left' if side == 0 else 'right'
        if force > peaks.get(key, {}).get('force_newtons', -math.inf):
            peaks[key] = dict(
                force_newtons=float(force), stamp_ns=int(stamp_ns), model=model,
                gripper_position=getattr(self, 'joints', {}).get(
                    'gripper_left_finger_joint'),
            )
        self._adaptive_force_peaks = peaks

    def _adaptive_close_for_grasp(
        self,
    ) -> Tuple[bool, float, bool, bool, bool]:
        """Run one overload-latched adaptive close attempt."""
        lock = getattr(self, '_lock', None)

        def set_monitor(active: bool) -> None:
            self._adaptive_close_active = active
            self._adaptive_overload_latched = None
            self._adaptive_effort_baseline_value = math.nan
            if active:
                self._held_grip_sensor_fault = None
                self._adaptive_force_peaks = {}
                self._adaptive_motion_halt_reason = None
                self._adaptive_hold_sent = False
            self._adaptive_motion_started = False

        with self._adaptive_command_guard():
            if lock is None:
                set_monitor(True)
            else:
                with lock:
                    set_monitor(True)
        result = None
        try:
            if _stock_diagnostic(self):
                controller = StockGripperClose(self)
                self._stock_close_controller = controller
                try:
                    result = controller.run()
                except Exception as exc:
                    self._hold_adaptive_gripper(str(exc))
                    raise
            elif getattr(self, 'fine_gripper_close_enabled', False):
                from .fine_gripper_close import FineGripperClose
                controller = FineGripperClose(self, self.fine_gripper_limits)
                self._fine_gripper_controller = controller
                try:
                    result = controller.run()
                except Exception:
                    controller._hold('fine_close_exception')
                    raise
            else:
                result = self._adaptive_close_attempt()
        finally:
            with self._adaptive_command_guard():
                stock_controller = getattr(self, '_stock_close_controller', None)
                if stock_controller is not None:
                    stock_controller.active = False
                    self._stock_close_controller = None
                controller = getattr(self, '_fine_gripper_controller', None)
                if controller is not None:
                    controller.active = False
                    self._fine_gripper_controller = None
                def finish_monitor():
                    # Snapshot and disarm in one sensor critical section. A
                    # callback must not insert a fault between those actions.
                    fault = (getattr(self, '_adaptive_motion_halt_reason', None)
                             or getattr(self, '_adaptive_overload_latched', None))
                    motion_started = getattr(self, '_adaptive_motion_started', False)
                    if fault:
                        self._transport_lock_engaged = False
                    if result is not None and result[0] and not fault:
                        self._acquired_effort_baseline = self._adaptive_effort_baseline_value
                    set_monitor(False)
                    return fault, motion_started

                if lock is None:
                    fault, motion_started = finish_monitor()
                else:
                    with lock:
                        fault, motion_started = finish_monitor()
                if fault:
                    if motion_started:
                        self._hold_adaptive_gripper(fault)
                    if result is not None:
                        result = (False, *result[1:])
        return result

    def _adaptive_close_attempt(
        self,
    ) -> Tuple[bool, float, bool, bool, bool]:
        """Close in small steps until sustained bilateral pressure is proven."""
        self._transport_lock_engaged = False
        self._clear_target_contact_samples()
        now_ns = int(self.get_clock().now().nanoseconds)
        baseline_effort = self._adaptive_effort_baseline(now_ns)
        if lock := getattr(self, '_lock', None):
            with lock:
                self._adaptive_effort_baseline_value = baseline_effort
        else:
            self._adaptive_effort_baseline_value = baseline_effort
        minimum_width = self.grasp_min_position + self.grasp_min_margin
        evidence, _, _ = self._adaptive_pressure_evidence(
            minimum_width=minimum_width,
            baseline_effort=baseline_effort,
        )
        feedback = self._latest_gripper_feedback()
        if feedback is None or not math.isfinite(feedback.position):
            return self._publish_adaptive_close_result(
                evidence,
                verified=False,
                reason='joint_feedback_unavailable',
                close_stage='adaptive_seek',
                commanded_position=math.nan,
                acquisition_position=None,
            )
        previous_position = float(feedback.position)
        if lock := getattr(self, '_lock', None):
            with lock:
                open_confirmed = bool(
                    getattr(self, '_gripper_open_confirmed', False)
                )
                self._gripper_open_confirmed = False
        else:
            open_confirmed = bool(
                getattr(self, '_gripper_open_confirmed', False)
            )
            self._gripper_open_confirmed = False
        if (
            not open_confirmed
            or abs(previous_position - self.gripper_open)
            > self.adaptive_start_tolerance
        ):
            return self._publish_adaptive_close_result(
                evidence,
                verified=False,
                reason='open_start_not_confirmed',
                close_stage='adaptive_seek',
                commanded_position=previous_position,
                acquisition_position=None,
            )
        if self._adaptive_evidence_is_terminal(evidence.reason):
            return self._publish_adaptive_close_result(
                evidence,
                verified=False,
                reason=evidence.reason,
                close_stage='adaptive_seek',
                commanded_position=previous_position,
                acquisition_position=None,
            )
        if not (
            minimum_width - self.adaptive_endpoint_tolerance
            <= previous_position
            <= self.gripper_open + self.adaptive_endpoint_tolerance
        ):
            return self._publish_adaptive_close_result(
                evidence,
                verified=False,
                reason='starting_width_out_of_bounds',
                close_stage='adaptive_seek',
                commanded_position=previous_position,
                acquisition_position=None,
            )

        first_unilateral_width: Optional[float] = None
        acquisition_position: Optional[float] = None
        commanded_position = previous_position
        failure_reason = 'bilateral_contact_not_found'

        while commanded_position > minimum_width + 1e-9:
            target = max(
                minimum_width,
                commanded_position - self.adaptive_close_step,
            )
            if not self._command_adaptive_gripper_step(target):
                failure_reason = (
                    getattr(self, '_adaptive_motion_halt_reason', None)
                    or self._adaptive_overload_reason()
                    or 'gripper_step_command_failed'
                )
                break
            commanded_position = target
            overload_reason = self._adaptive_overload_reason()
            if overload_reason is not None:
                failure_reason = overload_reason
                break
            evidence, model, identity_reason = (
                self._adaptive_pressure_evidence(
                    minimum_width=minimum_width,
                    baseline_effort=baseline_effort,
                )
            )
            feedback = self._latest_gripper_feedback()
            if feedback is None or not math.isfinite(feedback.position):
                failure_reason = 'joint_feedback_unavailable'
                break
            measured = float(feedback.position)
            if measured < target - self.adaptive_endpoint_tolerance:
                failure_reason = 'gripper_position_overshoot'
                break
            if measured > previous_position + self.adaptive_endpoint_tolerance:
                failure_reason = 'gripper_position_non_monotonic'
                break
            previous_position = measured
            if identity_reason in {
                'ambiguous_target_contact',
                'anonymous_target_contact',
            }:
                failure_reason = identity_reason
                break
            if (
                evidence.reason == 'width_too_wide'
                and (evidence.left_samples or evidence.right_samples)
            ):
                failure_reason = 'contact_above_grasp_width'
                break
            if self._adaptive_evidence_is_terminal(evidence.reason):
                failure_reason = evidence.reason
                break
            if evidence.verified:
                if model is None or identity_reason is not None:
                    failure_reason = identity_reason or 'target_not_latched'
                    break
                acquisition_position = float(evidence.width)
                failure_reason = ''
                break

            left_active = (
                evidence.left_samples > 0
                and evidence.left_force
                >= self.adaptive_contact_force_minimum
            )
            right_active = (
                evidence.right_samples > 0
                and evidence.right_force
                >= self.adaptive_contact_force_minimum
            )
            if left_active and right_active:
                if not self._wait_sim_duration(
                    self.adaptive_confirmation_seconds
                ):
                    failure_reason = 'contact_confirmation_timeout'
                    break
                evidence, model, identity_reason = (
                    self._adaptive_pressure_evidence(
                        minimum_width=minimum_width,
                        baseline_effort=baseline_effort,
                    )
                )
                if self._adaptive_evidence_is_terminal(evidence.reason):
                    failure_reason = evidence.reason
                    break
                if (
                    evidence.verified
                    and model is not None
                    and identity_reason is None
                ):
                    acquisition_position = float(evidence.width)
                    failure_reason = ''
                    break
                failure_reason = identity_reason or 'contact_not_stable'
                break
            if left_active != right_active:
                if first_unilateral_width is None:
                    first_unilateral_width = measured
                unilateral_travel = first_unilateral_width - measured
                if (
                    unilateral_travel
                    >= self.adaptive_unilateral_travel_limit
                    - self.adaptive_endpoint_tolerance
                ):
                    failure_reason = 'unilateral_contact_travel_limit'
                    break
            elif (
                measured > target + self.adaptive_endpoint_tolerance
            ):
                failure_reason = 'gripper_stalled_without_contact'
                break
            if target <= minimum_width + 1e-9:
                failure_reason = (
                    'unilateral_contact_at_width_floor'
                    if left_active != right_active
                    else 'bilateral_contact_not_found'
                )
                break

        if acquisition_position is None:
            return self._publish_adaptive_close_result(
                evidence,
                verified=False,
                reason=failure_reason,
                close_stage='adaptive_seek',
                commanded_position=commanded_position,
                acquisition_position=None,
            )

        preload_target = max(
            self.gripper_transport_lock,
            acquisition_position - self.adaptive_preload_distance,
        )
        if preload_target >= acquisition_position - 1e-9:
            return self._publish_adaptive_close_result(
                evidence,
                verified=False,
                reason='bounded_preload_unavailable',
                close_stage='adaptive_acquired',
                commanded_position=commanded_position,
                acquisition_position=acquisition_position,
            )
        if not self._command_adaptive_gripper_step(preload_target):
            return self._publish_adaptive_close_result(
                evidence,
                verified=False,
                reason=(getattr(self, '_adaptive_motion_halt_reason', None)
                        or self._adaptive_overload_reason() or 'preload_command_failed'),
                close_stage='adaptive_preload',
                commanded_position=preload_target,
                acquisition_position=acquisition_position,
            )
        overload_reason = self._adaptive_overload_reason()
        if overload_reason is not None:
            return self._publish_adaptive_close_result(
                evidence,
                verified=False,
                reason=overload_reason,
                close_stage='adaptive_preload',
                commanded_position=preload_target,
                acquisition_position=acquisition_position,
            )
        command_evidence, _, identity_reason = (
            self._adaptive_pressure_evidence(
                minimum_width=(
                    preload_target - self.adaptive_endpoint_tolerance
                ),
                baseline_effort=baseline_effort,
            )
        )
        if (
            self._adaptive_evidence_is_terminal(command_evidence.reason)
            or identity_reason in {
                'ambiguous_target_contact',
                'anonymous_target_contact',
            }
        ):
            return self._publish_adaptive_close_result(
                command_evidence,
                verified=False,
                reason=identity_reason or command_evidence.reason,
                close_stage='adaptive_preload',
                commanded_position=preload_target,
                acquisition_position=acquisition_position,
            )

        # Command-phase callbacks cannot certify the hold.  Reset the epoch,
        # keep the concrete target identity, and demand a new pressure window.
        self._clear_target_contact_samples()
        if not self._wait_sim_duration(self.adaptive_confirmation_seconds):
            return self._publish_adaptive_close_result(
                command_evidence,
                verified=False,
                reason='post_preload_confirmation_timeout',
                close_stage='adaptive_preload',
                commanded_position=preload_target,
                acquisition_position=acquisition_position,
            )
        overload_reason = self._adaptive_overload_reason()
        if overload_reason is not None:
            return self._publish_adaptive_close_result(
                command_evidence,
                verified=False,
                reason=overload_reason,
                close_stage='adaptive_preload',
                commanded_position=preload_target,
                acquisition_position=acquisition_position,
            )
        evidence, model, identity_reason = self._adaptive_pressure_evidence(
            minimum_width=(
                preload_target - self.adaptive_endpoint_tolerance
            ),
            baseline_effort=baseline_effort,
        )
        verified = bool(
            evidence.verified
            and model is not None
            and identity_reason is None
        )
        return self._publish_adaptive_close_result(
            evidence,
            verified=verified,
            reason=(
                'bilateral_pressure_confirmed'
                if verified
                else identity_reason or evidence.reason
            ),
            close_stage='transport_lock' if verified else 'adaptive_preload',
            commanded_position=preload_target,
            acquisition_position=acquisition_position,
        )

    def _close_for_grasp(self) -> Tuple[bool, float, bool, bool, bool]:
        """Run adaptive pressure close in production and legacy test fixtures."""
        if hasattr(self, 'adaptive_close_step'):
            return self._adaptive_close_for_grasp()
        return self._legacy_staged_close_for_grasp()

    def _legacy_staged_close_for_grasp(
        self,
    ) -> Tuple[bool, float, bool, bool, bool]:
        """Stop at the first bilateral, non-empty stage of a guarded close."""
        self._transport_lock_engaged = False
        self._clear_target_contact_samples()
        if not self._command_gripper(self.gripper_preclose):
            return False, 0.0, False, False, False
        sample = self._pinch_sample()
        close_stage = 'preclose'
        # A command equal to the measured first-contact width produces
        # essentially no squeeze.  Move a bounded distance farther inward,
        # while staying above the empty-grasp threshold, then re-sample both
        # target contacts and the measured width before any loaded motion.
        minimum_plausible_width = self.grasp_min_position + self.grasp_min_margin
        if sample[1] >= minimum_plausible_width:
            if not self._command_gripper(self.gripper_preload):
                return False, 0.0, False, False, False
            sample = self._pinch_sample()
            close_stage = 'preload'
        acquisition_position = float(sample[1])
        acquisition_plausible_width = bool(sample[4])
        if sample[0]:
            # The lower lock target must never acquire a book by itself.  Arm
            # it only after the ordinary-width preload proved a fresh bilateral
            # pinch.  Once the uninterrupted lock command has settled, clear
            # every command-phase callback and demand a new bilateral target
            # contact sample before accepting the narrower retention band.
            if not self._command_gripper(self.gripper_transport_lock):
                return False, 0.0, False, False, False
            self._transport_lock_engaged = True
            lock_verified = self._fresh_retention_probe(
                'pick',
                'transport_lock',
            )
            sample = self._pinch_sample()
            lock_verified = bool(lock_verified and sample[0])
            sample = (lock_verified, *sample[1:])
            if not lock_verified:
                self._transport_lock_engaged = False
            close_stage = 'transport_lock'
        verified, grasp_width, left_contact, right_contact, plausible_width = sample
        self._publish_status(
            'gripper_closed',
            close_stage=close_stage,
            commanded_position=(
                self.gripper_transport_lock
                if close_stage == 'transport_lock'
                else self.gripper_preload
                if close_stage == 'preload'
                else self.gripper_preclose
            ),
            acquisition_position=acquisition_position,
            acquisition_plausible_width=acquisition_plausible_width,
            transport_lock_engaged=bool(
                getattr(self, '_transport_lock_engaged', False)
            ),
            measured_position=grasp_width,
            left_contact=left_contact,
            right_contact=right_contact,
            bilateral_contact=left_contact and right_contact,
            plausible_width=plausible_width,
        )
        return verified, grasp_width, left_contact, right_contact, plausible_width

    def _move_ik(self, solution: Sequence[float], duration: float = 3.0) -> bool:
        q = np.asarray(solution, dtype=float)
        if q.shape != (8,):
            raise ValueError('IK solution must contain torso plus seven arm joints')
        torso_ok = self._follow(
            self.torso_client, ['torso_lift_joint'], [q[0]], min(2.5, duration)
        )
        if not torso_ok:
            return False
        return self._follow(self.arm_client, ARM_JOINTS, q[1:], duration)

    def _stow(self, *, initial_stow_serial=None) -> bool:
        if getattr(self, '_empty_arm_staged', False):
            raise RuntimeError('staged empty arm requires a checked recovery before stow')
        if initial_stow_serial is not None:
            return _initial_stow_serial.run_stow(self, initial_stow_serial)
        # Park the unused arm once so its zero-position wrist cannot contact
        # the shelf.  All task manipulation remains exclusively left-armed.
        if not self._open_gripper():
            return False
        if not self._move_ik(HOME, 3.5):
            return False
        if self._right_parked:
            return True
        self._right_parked = self._follow(
            self.right_arm_client,
            RIGHT_ARM_JOINTS,
            RIGHT_HOME,
            3.5,
        )
        return self._right_parked

    def _point_in_base(self, message: Optional[PointStamped]) -> np.ndarray:
        if message is None:
            raise RuntimeError('no fresh perception point is available')
        measurement_time = Time.from_msg(message.header.stamp)
        age = (self.get_clock().now() - measurement_time).nanoseconds / 1e9
        if age < -0.10 or age > self.perception_max_age:
            raise RuntimeError(f'perception point is stale (age={age:.3f}s)')
        try:
            transformed = self.tf_buffer.transform(
                message,
                'base_footprint',
                timeout=Duration(seconds=self.tf_timeout),
            )
        except TransformException as exc:
            raise RuntimeError(f'cannot transform perception point: {exc}') from exc
        return np.asarray(
            [transformed.point.x, transformed.point.y, transformed.point.z], dtype=float
        )

    def _wait_for_perception_point(self, attribute: str) -> np.ndarray:
        deadline = time.monotonic() + self.perception_wait
        last_error = 'no fresh perception point is available'
        while not self._cancel.is_set() and time.monotonic() <= deadline:
            try:
                if attribute == 'latest_bin' and not bin_message_is_current(
                    self.latest_bin, self.get_clock().now().nanoseconds,
                    getattr(self, 'bin_invalidated_ns', -1),
                ):
                    raise RuntimeError('no fresh verified bin point is available')
                message = getattr(self, attribute)
                point = self._point_in_base(message)
                if attribute == 'latest_bin' and getattr(self, 'table_scene_required', False):
                    stamp_ns = int(message.header.stamp.sec)*1_000_000_000 + int(message.header.stamp.nanosec)
                    entry = getattr(self, '_latest_table_scene_entry', None)
                    reference = measured_scene_context(self)
                    if getattr(self, 'bin_scene_required', False):
                        table, registered_bin = match_registered_scenes(entry, stamp_ns, point, reference)
                        self._selected_place_table_scene = table
                        self._selected_place_bin_scene = registered_bin
                        # Navigation and reacquisition keep the original point;
                        # only this admitted PLACE input becomes the CAD center.
                        point = np.asarray(registered_bin['floor_center'], dtype=float)
                    else:
                        self._selected_place_table_scene = matching_table_scene(entry, stamp_ns, point)
                    self._selected_place_scene_reference = reference
                return point
            except RuntimeError as exc:
                last_error = str(exc)
                time.sleep(0.05)
        raise RuntimeError(last_error)

    def _current_seed(self) -> np.ndarray:
        lock = getattr(self, '_lock', None)
        if lock is None:
            joints = dict(self.joints)
        else:
            with lock:
                joints = dict(self.joints)
        if all(name in joints for name in IK_JOINTS):
            return np.asarray([joints[name] for name in IK_JOINTS], dtype=float)
        return OFFER.copy()

    def _measured_left_solution(self) -> np.ndarray:
        """Return the measured torso/left-arm state or fail closed."""
        lock = getattr(self, '_lock', None)
        if lock is None:
            joints = dict(self.joints)
        else:
            with lock:
                joints = dict(self.joints)
        return np.asarray(
            measured_joint_positions(
                joints,
                IK_JOINTS,
                group='torso and left arm',
            ),
            dtype=float,
        )

    @staticmethod
    def _interpolate_positions(
        start: Sequence[float],
        end: Sequence[float],
        maximum_step: float,
    ) -> List[np.ndarray]:
        first = np.asarray(start, dtype=float)
        last = np.asarray(end, dtype=float)
        distance = float(np.linalg.norm(last - first))
        steps = max(1, int(np.ceil(distance / maximum_step)))
        return [
            first + (last - first) * (index / steps)
            for index in range(1, steps + 1)
        ]

    def _retracted_transition_is_safe(
        self,
        start: Sequence[float],
        end: Sequence[float],
    ) -> bool:
        first = np.asarray(start, dtype=float)
        last = np.asarray(end, dtype=float)
        maximum_x = self.cartesian_clearance + 0.20
        maximum_abs_y = 0.55
        for fraction in np.linspace(0.0, 1.0, 25):
            q = first + (last - first) * fraction
            points = self.chain.link_positions(q)
            if (
                float(np.max(points[:, 0])) > maximum_x
                or float(np.max(np.abs(points[:, 1]))) > maximum_abs_y
            ):
                return False
        return True

    def _attached_book_corners(
        self,
        front: Sequence[float],
        grasp_solution: Sequence[float],
    ) -> np.ndarray:
        """Return an inflated official book box in the grasp-link frame."""
        front_point = np.asarray(front, dtype=float)
        if front_point.shape != (3,):
            raise ValueError('book front point must contain x, y, and z')
        half_extents = 0.5 * self.carried_book_dimensions
        centre = front_point + np.asarray([half_extents[0], 0.0, 0.0])
        inflated = half_extents + self.carried_book_padding
        signs = np.asarray(
            [
                (x_sign, y_sign, z_sign)
                for x_sign in (-1.0, 1.0)
                for y_sign in (-1.0, 1.0)
                for z_sign in (-1.0, 1.0)
            ],
            dtype=float,
        )
        world_corners = centre + signs * inflated
        grasp_transform = self.chain.forward(grasp_solution)
        # For row vectors, inverse rigid rotation is multiplication by R.
        return (
            world_corners - grasp_transform[:3, 3]
        ) @ grasp_transform[:3, :3]

    def _carried_transition_is_safe(
        self,
        start: Sequence[float],
        end: Sequence[float],
        attached_corners: Sequence[Sequence[float]],
        shelf_front_x: float,
    ) -> bool:
        """Check robot links and the inflated carried-book sweep at the shelf."""
        if not self._retracted_transition_is_safe(start, end):
            return False
        return self._carried_volume_transition_is_safe(
            start,
            end,
            attached_corners,
            maximum_payload_x=(
                float(shelf_front_x) - self.carried_shelf_margin
            ),
        )

    def _carried_robot_transition_is_safe(
        self,
        start: Sequence[float],
        end: Sequence[float],
        attached_corners: Sequence[Sequence[float]],
        *,
        right_positions: Optional[Sequence[float]] = None,
        head_positions: Optional[Sequence[float]] = None,
    ) -> bool:
        """Check a carried-book sweep against robot collision geometry."""
        context = {}
        if right_positions is not None:
            context['right_positions'] = right_positions
        if head_positions is not None:
            context['head_positions'] = head_positions
        return self._carried_volume_transition_is_safe(
            start,
            end,
            attached_corners,
            **context,
        )

    def _gravity_supported_transition_is_safe(
        self,
        start: Sequence[float],
        end: Sequence[float],
    ) -> bool:
        """Require the lower finger to support the payload for a whole leg."""
        first = np.asarray(start, dtype=float)
        last = np.asarray(end, dtype=float)
        unchanged = np.allclose(first, last, rtol=0.0, atol=1e-12)
        fractions = (
            (0.0,)
            if unchanged
            else np.linspace(0.0, 1.0, self.carried_transition_samples)
        )
        return all(
            float(self.chain.forward(first + (last - first) * fraction)[2, 1])
            >= self.carried_supported_jaw_vertical_component
            for fraction in fractions
        )

    def _carried_post_retreat_transition_is_safe(
        self,
        start: Sequence[float],
        end: Sequence[float],
        attached_corners: Sequence[Sequence[float]],
        shelf_front_x: float,
    ) -> bool:
        """Check the payload and full modeled robot against the retreated shelf."""
        maximum_x = float(shelf_front_x) - self.carried_shelf_margin
        return self._carried_volume_transition_is_safe(
            start,
            end,
            attached_corners,
            maximum_payload_x=maximum_x,
            maximum_robot_x=maximum_x,
        )

    def _plan_carried_joint_route(
        self,
        start: Sequence[float],
        goals: Sequence[Sequence[float]],
        attached_corners: Sequence[Sequence[float]],
        *,
        shelf_front_x: Optional[float] = None,
        post_retreat_shelf_front_x: Optional[float] = None,
        require_gravity_support: bool = False,
        geometry_backend=None,
    ) -> Optional[List[np.ndarray]]:
        """Expand and validate controller-sized carried joint-space legs."""
        if shelf_front_x is not None and post_retreat_shelf_front_x is not None:
            raise ValueError('only one shelf-front constraint may be supplied')
        if geometry_backend is not None and shelf_front_x is not None:
            raise ValueError('parallel carry supports post-retreat shelf constraints only')
        route: List[np.ndarray] = []
        previous = np.asarray(start, dtype=float)
        for goal in goals:
            for waypoint in interpolate_joint_waypoints(
                previous,
                goal,
                first_arm_index=1,
                maximum_joint_step=self.cartesian_joint_step,
                orientation_distance=lambda first, last: np.linalg.norm(
                    self.chain.pose_error(
                        self.chain.forward(first),
                        self.chain.forward(last),
                    )[3:]
                ),
                maximum_orientation_step=self.carried_orientation_step_limit,
            ):
                if geometry_backend is not None:
                    constraints = {}
                    if post_retreat_shelf_front_x is not None:
                        maximum_x = float(post_retreat_shelf_front_x) - self.carried_shelf_margin
                        constraints = dict(maximum_payload_x=maximum_x, maximum_robot_x=maximum_x)
                    safe = geometry_backend.volume(previous, waypoint, attached_corners, **constraints)
                elif post_retreat_shelf_front_x is not None:
                    safe = self._carried_post_retreat_transition_is_safe(
                        previous,
                        waypoint,
                        attached_corners,
                        post_retreat_shelf_front_x,
                    )
                elif shelf_front_x is None:
                    safe = self._carried_robot_transition_is_safe(
                        previous,
                        waypoint,
                        attached_corners,
                    )
                else:
                    safe = self._carried_transition_is_safe(
                        previous,
                        waypoint,
                        attached_corners,
                        shelf_front_x,
                    )
                if not safe:
                    return None
                if require_gravity_support and not (
                    self._gravity_supported_transition_is_safe(previous, waypoint)
                ):
                    return None
                route.append(waypoint)
                previous = waypoint
        return route

    def _carried_volume_transition_is_safe(
        self,
        start: Sequence[float],
        end: Sequence[float],
        attached_corners: Sequence[Sequence[float]],
        *,
        maximum_payload_x: Optional[float] = None,
        maximum_robot_x: Optional[float] = None,
        right_positions: Optional[Sequence[float]] = None,
        head_positions: Optional[Sequence[float]] = None,
    ) -> bool:
        context = {}
        if right_positions is not None:
            context['right_positions'] = self._resolved_right_positions(right_positions)
        if head_positions is not None:
            context['head_positions'] = self._resolved_head_positions(head_positions)
        first = np.asarray(start, dtype=float)
        last = np.asarray(end, dtype=float)
        if float(np.max(np.abs(last[1:] - first[1:]))) > self.cartesian_joint_step:
            return False
        orientation_delta = self.chain.pose_error(
            self.chain.forward(first),
            self.chain.forward(last),
        )[3:]
        if float(np.linalg.norm(orientation_delta)) > self.carried_orientation_step_limit:
            return False
        local_corners = np.asarray(attached_corners, dtype=float)
        if local_corners.shape != (8, 3):
            raise ValueError('attached book envelope must contain eight 3-D corners')
        local_vertical = local_corners[1] - local_corners[0]
        vertical_norm = float(np.linalg.norm(local_vertical))
        if vertical_norm <= 1e-9:
            raise ValueError('attached book envelope has no vertical axis')
        local_vertical /= vertical_norm
        minimum_vertical_component = float(np.cos(self.carried_maximum_tilt))
        unchanged = np.allclose(first, last, rtol=0.0, atol=1e-12)
        fractions = (
            (0.0,)
            if unchanged
            else np.linspace(0.0, 1.0, self.carried_transition_samples)
        )
        for fraction in fractions:
            if getattr(self, '_cancel', None) is not None and self._cancel.is_set():
                return False
            q = first + (last - first) * fraction
            transform = self.chain.forward(q)
            book_vertical_component = float(
                (transform[:3, :3] @ local_vertical)[2]
            )
            jaw_vertical_component = abs(float(transform[2, 1]))
            if (
                book_vertical_component < minimum_vertical_component
                and jaw_vertical_component
                < self.carried_supported_jaw_vertical_component
            ):
                return False
            world_corners = (
                local_corners @ transform[:3, :3].T
                + transform[:3, 3]
            )
            if (
                maximum_payload_x is not None
                and float(np.max(world_corners[:, 0])) > maximum_payload_x
            ):
                return False
            if self._carried_robot_collision(
                q,
                world_corners,
                maximum_robot_x=maximum_robot_x,
                **context,
            ) is not None:
                return False
        # Reuse the already bounded leg, but refine it once more for exact
        # robot self-collision checks.  This adds an interior check to every
        # non-zero carried move and adaptively adds more when either joint or
        # grasp-link attitude change would exceed half of the controller-leg
        # limits.  Keep this separate from the denser payload sweep above so
        # the expensive mesh/mesh test does not run at every payload sample.
        self_collision_waypoints = list(
            interpolate_joint_waypoints(
                first,
                last,
                first_arm_index=1,
                maximum_joint_step=0.5 * self.cartesian_joint_step,
                orientation_distance=lambda start_q, end_q: np.linalg.norm(
                    self.chain.pose_error(
                        self.chain.forward(start_q),
                        self.chain.forward(end_q),
                    )[3:]
                ),
                maximum_orientation_step=(
                    0.5 * self.carried_orientation_step_limit
                ),
            )
        )
        if (
            len(self_collision_waypoints) == 1
            and not unchanged
        ):
            self_collision_waypoints.insert(0, 0.5 * (first + last))
        for q in (first, *self_collision_waypoints):
            if getattr(self, '_cancel', None) is not None and self._cancel.is_set():
                return False
            if self._robot_self_collision(q, **context) is not None:
                return False
        return True

    def _collision_link_transforms(
        self,
        solution: Sequence[float],
        *,
        right_positions: Optional[Sequence[float]] = None,
        head_positions: Optional[Sequence[float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Return moving-left, parked-right, and measured-head transforms."""
        q = np.asarray(solution, dtype=float)
        transforms = self.chain.link_transforms(q)
        resolved_right = self._resolved_right_positions(right_positions)
        right_q = np.concatenate(([q[0]], resolved_right))
        right_transforms = self.right_chain.link_transforms(right_q)
        transforms.update(
            {
                link: right_transforms[link]
                for link in RIGHT_COLLISION_LINKS
            }
        )
        head_q = np.concatenate(
            ([q[0]], self._resolved_head_positions(head_positions))
        )
        head_transforms = self.head_chain.link_transforms(head_q)
        transforms.update(
            {link: head_transforms[link] for link in HEAD_COLLISION_LINKS}
        )
        return transforms

    def _right_joint_positions(self) -> np.ndarray:
        """Return the measured parked-arm joints or fail closed."""
        return np.asarray(
            measured_joint_positions(
                self.joints,
                RIGHT_ARM_JOINTS,
                group='right arm',
            ),
            dtype=float,
        )

    def _resolved_right_positions(
        self,
        positions: Optional[Sequence[float]],
    ) -> np.ndarray:
        result = (
            self._right_joint_positions()
            if positions is None
            else np.asarray(positions, dtype=float)
        )
        if (
            result.shape != (len(RIGHT_ARM_JOINTS),)
            or not np.all(np.isfinite(result))
        ):
            raise ValueError(
                'right-arm collision state must contain seven finite joints'
            )
        return result

    def _head_joint_positions(self) -> np.ndarray:
        """Return measured head joints, failing closed when state is absent."""
        missing = [name for name in HEAD_JOINTS if name not in self.joints]
        if missing:
            raise RuntimeError(
                f'Cannot collision-check without head joints: {missing}'
            )
        return np.asarray([float(self.joints[name]) for name in HEAD_JOINTS])

    def _resolved_head_positions(
        self,
        positions: Optional[Sequence[float]],
    ) -> np.ndarray:
        result = (
            self._head_joint_positions()
            if positions is None
            else np.asarray(positions, dtype=float)
        )
        if result.shape != (len(HEAD_JOINTS),) or not np.all(np.isfinite(result)):
            raise ValueError('head collision state must contain two finite joints')
        return result

    def _world_collision_surfaces(
        self,
        solution: Sequence[float],
        *,
        right_positions: Optional[Sequence[float]] = None,
        head_positions: Optional[Sequence[float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Transform and group official robot collision facets by link."""
        transforms = self._collision_link_transforms(
            solution,
            right_positions=right_positions,
            head_positions=head_positions,
        )
        grouped: Dict[str, List[np.ndarray]] = {}
        for collision_mesh in self.carried_collision_meshes:
            transform = transforms[collision_mesh.link]
            world_triangles = (
                collision_mesh.triangles @ transform[:3, :3].T
                + transform[:3, 3]
            )
            grouped.setdefault(collision_mesh.link, []).append(world_triangles)
        return {
            link: meshes[0] if len(meshes) == 1 else np.concatenate(meshes)
            for link, meshes in grouped.items()
        }

    def _watertight_collision_links(self) -> frozenset[str]:
        """Return links whose complete collision surface is verified closed."""
        closed_by_link: Dict[str, bool] = {}
        for collision_mesh in getattr(self, 'carried_collision_meshes', ()):
            closed_by_link[collision_mesh.link] = (
                closed_by_link.get(collision_mesh.link, True)
                and collision_mesh.watertight
            )
        return frozenset(
            link for link, closed in closed_by_link.items() if closed
        )

    def _robot_self_collision(
        self,
        solution: Sequence[float],
        collision_surfaces: Optional[Dict[str, np.ndarray]] = None,
        *,
        head_positions: Optional[Sequence[float]] = None,
        right_positions: Optional[Sequence[float]] = None,
        _scene_snapshot=None,
    ) -> Optional[Tuple[str, str]]:
        """Return the first intersecting pair of nonadjacent robot links."""
        q = np.asarray(solution, dtype=float)
        right_positions = self._resolved_right_positions(right_positions)
        resolved_head = self._resolved_head_positions(head_positions)
        cache_key = joint_state_cache_key(
            q,
            right_positions,
            resolved_head,
        )
        if collision_surfaces is None and cache_key in self._self_collision_cache:
            return self._self_collision_cache[cache_key]
        shared_geometry = None
        if collision_surfaces is None and _scene_snapshot is not None:
            from erc_phase1_solution.sample_collision_snapshot import _SceneRobotSnapshot
            if (type(_scene_snapshot) is _SceneRobotSnapshot
                    and getattr(self._world_collision_surfaces, '__func__', None)
                        is ManipulationNode._world_collision_surfaces
                    and getattr(self._collision_link_transforms, '__func__', None)
                        is ManipulationNode._collision_link_transforms):
                shared_geometry = _scene_snapshot.resolve(self, q, right_positions, resolved_head)
        surfaces = (
            self._world_collision_surfaces(
                q,
                right_positions=right_positions,
                head_positions=resolved_head,
            )
            if collision_surfaces is None
            else collision_surfaces
        ) if shared_geometry is None else shared_geometry[0]
        links = [link for link in CARRIED_COLLISION_LINKS if link in surfaces]
        watertight_links = self._watertight_collision_links()
        bounds = {
            link: np.asarray(
                [
                    np.min(surface, axis=(0, 1)),
                    np.max(surface, axis=(0, 1)),
                ]
            )
            for link, surface in surfaces.items()
        } if shared_geometry is None else shared_geometry[1]
        prepared_surfaces = {}

        def links_intersect(first_link: str, second_link: str) -> bool:
            first_bounds = bounds[first_link]
            second_bounds = bounds[second_link]
            if np.any(first_bounds[1] < second_bounds[0]) or np.any(
                second_bounds[1] < first_bounds[0]
            ):
                return False
            if collision_surfaces is None:
                separator = getattr(self, '_static_pair_separation', None)
                if separator is None:
                    from erc_phase1_solution.static_pair_separation import StaticPairSeparation
                    separator = StaticPairSeparation()
                    self._static_pair_separation = separator
                if separator.separated(
                    first_link, second_link,
                    surfaces[first_link], surfaces[second_link],
                ):
                    return False
            for link in (first_link, second_link):
                if link not in prepared_surfaces:
                    prepared_surfaces[link] = PreparedTriangleMesh(surfaces[link])
            return triangle_meshes_intersect(
                prepared_surfaces[first_link],
                prepared_surfaces[second_link],
                first_watertight=first_link in watertight_links,
                second_watertight=second_link in watertight_links,
            )

        static_links = [
            link for link in links if link not in LEFT_COLLISION_LINKS
        ]
        static_cache_key = joint_state_cache_key(
            (q[0],),
            right_positions,
            resolved_head,
        )
        static_result = None
        if collision_surfaces is None:
            static_result = self._static_self_collision_cache.get(
                static_cache_key
            )
        if static_result is None and (
            collision_surfaces is not None
            or static_cache_key not in self._static_self_collision_cache
        ):
            for first_index, first_link in enumerate(static_links):
                for second_link in static_links[first_index + 1:]:
                    if frozenset((first_link, second_link)) in (
                        DIRECTLY_CONNECTED_COLLISION_LINKS
                    ):
                        continue
                    if links_intersect(first_link, second_link):
                        static_result = (first_link, second_link)
                        break
                if static_result is not None:
                    break
            if collision_surfaces is None:
                self._static_self_collision_cache[
                    static_cache_key
                ] = static_result
        if static_result is not None:
            if collision_surfaces is None:
                self._self_collision_cache[cache_key] = static_result
            return static_result

        for first_index, first_link in enumerate(links):
            for second_link in links[first_index + 1:]:
                if (
                    first_link not in LEFT_COLLISION_LINKS
                    and second_link not in LEFT_COLLISION_LINKS
                ):
                    continue
                if frozenset((first_link, second_link)) in (
                    DIRECTLY_CONNECTED_COLLISION_LINKS
                ):
                    continue
                if links_intersect(first_link, second_link):
                    result = (first_link, second_link)
                    if collision_surfaces is None:
                        self._self_collision_cache[cache_key] = result
                    return result
        if collision_surfaces is None:
            self._self_collision_cache[cache_key] = None
        return None

    # Bind the original consumer once. An instance/subclass override keeps its
    # original call signature unless it deliberately opts into this protocol.
    _scene_snapshot_body = _robot_self_collision

    def _carried_robot_collision(
        self,
        solution: Sequence[float],
        world_corners: Sequence[Sequence[float]],
        *,
        head_positions: Optional[Sequence[float]] = None,
        maximum_robot_x: Optional[float] = None,
        right_positions: Optional[Sequence[float]] = None,
    ) -> Optional[str]:
        """Return the first robot link intersecting the inflated carried book."""
        transforms = self._collision_link_transforms(
            solution,
            head_positions=head_positions,
            **({'right_positions': right_positions} if right_positions is not None else {}),
        )
        book_bounds = np.asarray(
            [
                np.min(world_corners, axis=0),
                np.max(world_corners, axis=0),
            ]
        )
        for collision_mesh in self.carried_collision_meshes:
            transform = transforms[collision_mesh.link]
            local_center = np.mean(collision_mesh.bounds, axis=0)
            local_half_extents = 0.5 * (
                collision_mesh.bounds[1] - collision_mesh.bounds[0]
            )
            world_center = (
                transform[:3, :3] @ local_center + transform[:3, 3]
            )
            world_half_extents = (
                np.abs(transform[:3, :3]) @ local_half_extents
            )
            if (
                maximum_robot_x is not None
                and float(world_center[0] + world_half_extents[0])
                > maximum_robot_x
            ):
                return collision_mesh.link
            if np.any(
                world_center + world_half_extents < book_bounds[0]
            ) or np.any(
                world_center - world_half_extents > book_bounds[1]
            ):
                continue
            world_triangles = (
                collision_mesh.triangles @ transform[:3, :3].T
                + transform[:3, 3]
            )
            if oriented_box_intersects_triangles(
                world_corners,
                world_triangles,
                closed_surface=collision_mesh.watertight,
            ):
                return collision_mesh.link
        return None

    def _head_motion_collision(
        self,
        solution: Sequence[float],
        head_positions: Sequence[float],
        world_corners: Sequence[Sequence[float]],
        *,
        right_positions: Optional[Sequence[float]] = None,
    ) -> Optional[str]:
        """Return a payload or robot collision caused by one head posture."""
        resolved_head = self._resolved_head_positions(head_positions)
        surfaces = self._world_collision_surfaces(
            solution,
            head_positions=resolved_head,
            **({'right_positions': right_positions} if right_positions is not None else {}),
        )
        watertight_links = self._watertight_collision_links()
        bounds = {
            link: np.asarray(
                [
                    np.min(surface, axis=(0, 1)),
                    np.max(surface, axis=(0, 1)),
                ]
            )
            for link, surface in surfaces.items()
        }
        links = [link for link in CARRIED_COLLISION_LINKS if link in surfaces]
        for first_index, first_link in enumerate(links):
            for second_link in links[first_index + 1:]:
                if (
                    first_link not in HEAD_COLLISION_LINKS
                    and second_link not in HEAD_COLLISION_LINKS
                ):
                    continue
                if frozenset((first_link, second_link)) in (
                    DIRECTLY_CONNECTED_COLLISION_LINKS
                ):
                    continue
                first_bounds = bounds[first_link]
                second_bounds = bounds[second_link]
                if np.any(first_bounds[1] < second_bounds[0]) or np.any(
                    second_bounds[1] < first_bounds[0]
                ):
                    continue
                if triangle_meshes_intersect(
                    surfaces[first_link],
                    surfaces[second_link],
                    first_watertight=first_link in watertight_links,
                    second_watertight=second_link in watertight_links,
                ):
                    return f'{first_link}:{second_link}'

        corners = np.asarray(world_corners, dtype=float)
        for link in HEAD_COLLISION_LINKS:
            link_meshes = tuple(
                mesh for mesh in self.carried_collision_meshes
                if mesh.link == link
            )
            if oriented_box_intersects_triangles(
                corners,
                surfaces[link],
                closed_surface=all(mesh.watertight for mesh in link_meshes),
            ):
                return f'payload:{link}'
        return None

    def _carried_head_transition_is_safe(
        self,
        pan: float,
        tilt: float,
    ) -> bool:
        """Preflight a measured-to-commanded head sweep around held cargo."""
        if self._held_book_corners is None:
            return True
        target = np.asarray([pan, tilt], dtype=float)
        if not np.all(np.isfinite(target)):
            return False
        if np.any(target < self.head_chain.lower[1:]) or np.any(
            target > self.head_chain.upper[1:]
        ):
            return False
        start = self._head_joint_positions()
        solution = self._current_seed()
        grasp_transform = self.chain.forward(solution)
        world_corners = (
            self._held_book_corners @ grasp_transform[:3, :3].T
            + grasp_transform[:3, 3]
        )
        # Establish that the unchanged arm/body configuration is valid once;
        # subsequent samples need only recheck geometry involving the head.
        if (
            self._robot_self_collision(solution, head_positions=start) is not None
            or self._carried_robot_collision(
                solution,
                world_corners,
                head_positions=start,
            ) is not None
        ):
            return False
        for fraction in np.linspace(
            0.0,
            1.0,
            self.carried_transition_samples,
        ):
            sample = start + (target - start) * fraction
            if self._head_motion_collision(
                solution,
                sample,
                world_corners,
            ) is not None:
                return False
        return True

    def _carried_navigation_radius(
        self,
        solution: Sequence[float],
        attached_corners: Sequence[Sequence[float]],
    ) -> float:
        """Return the arm/base/payload planar radius about base_footprint."""
        q = np.asarray(solution, dtype=float)
        transforms = self._collision_link_transforms(q)
        grasp_transform = self.chain.forward(q)
        local_corners = np.asarray(attached_corners, dtype=float)
        world_corners = (
            local_corners @ grasp_transform[:3, :3].T
            + grasp_transform[:3, 3]
        )
        maximum_radius = float(
            np.max(np.linalg.norm(world_corners[:, :2], axis=1))
        )
        for collision_mesh in self.carried_collision_meshes:
            transform = transforms[collision_mesh.link]
            world_vertices = (
                collision_mesh.triangles.reshape((-1, 3))
                @ transform[:3, :3].T
                + transform[:3, 3]
            )
            maximum_radius = max(
                maximum_radius,
                float(np.max(np.linalg.norm(world_vertices[:, :2], axis=1))),
            )
        return maximum_radius

    def _supported_compact_goals(
        self,
        start: Sequence[float],
        *,
        shoulder_progress_power: float = 1.0,
    ) -> List[np.ndarray]:
        """Coordinate proximal tuck and wrist roll without losing support."""
        if not 0.5 <= shoulder_progress_power <= 1.0:
            raise ValueError('shoulder progress power must be within [0.5, 1.0]')
        first = np.asarray(start, dtype=float)
        goal = SUPPORTED_CARRY.copy()
        goal[0] = first[0]
        # Twenty proximal samples keep the compensated wrist solution smooth
        # enough that the dense collision/orientation validator accepts every
        # swept segment.  For each step, choose the q7 angle that maximizes the
        # world-vertical component of the closing axis.  The lower search bound
        # preserves 0.27 rad of margin to the official soft joint limit.
        goals: List[np.ndarray] = []
        previous_q7 = float(first[-1])
        q7_lower = max(
            float(self.chain.lower[-1]) + 0.01,
            float(SUPPORTED_CARRY[-1]),
        )
        # Stay on the negative q7 branch established by the top-row cradle.
        # The equally supported positive branch is disconnected by a roughly
        # pi-radian wrist jump and is not a valid continuation of this grasp.
        q7_upper = min(0.0, float(self.chain.upper[-1]) - 0.01)
        for fraction in np.linspace(0.05, 1.0, 20):
            candidate = first + (goal - first) * fraction
            # A shallow grasp extends the held book farther beyond the palm.
            # Turning the shoulder earlier can clear the shelf during the
            # tuck; the endpoint and all swept-volume checks stay unchanged.
            candidate[1] = first[1] + (goal[1] - first[1]) * (
                fraction ** shoulder_progress_power
            )
            zero_roll = candidate.copy()
            zero_roll[-1] = 0.0
            quarter_roll = zero_roll.copy()
            quarter_roll[-1] = 0.5 * np.pi
            vertical_at_zero = float(self.chain.forward(zero_roll)[2, 1])
            vertical_at_quarter = float(
                self.chain.forward(quarter_roll)[2, 1]
            )
            optimum = float(
                np.arctan2(vertical_at_quarter, vertical_at_zero)
            )
            q7_candidates = [q7_lower, q7_upper]
            for period in range(-3, 4):
                for angle in (optimum + period * np.pi,):
                    if q7_lower <= angle <= q7_upper:
                        q7_candidates.append(float(angle))
            scored = []
            for q7 in q7_candidates:
                sample = candidate.copy()
                sample[-1] = q7
                support = abs(float(self.chain.forward(sample)[2, 1]))
                scored.append((support, q7))
            maximum_support = max(support for support, _ in scored)
            near_maximum = [
                q7
                for support, q7 in scored
                if maximum_support - support <= 1e-9
            ]
            selected_q7 = min(
                near_maximum,
                key=lambda q7: (abs(q7 - previous_q7), q7),
            )
            candidate[-1] = selected_q7
            if abs(float(self.chain.forward(candidate)[2, 1])) < (
                self.carried_supported_jaw_vertical_component
            ):
                raise RuntimeError(
                    'No gravity-supported wrist angle for compact transport'
                )
            goals.append(candidate)
            previous_q7 = selected_q7
        return goals

    def _solve_fixed_orientation_lowering(
        self,
        start_solution: Sequence[float],
        rotation: np.ndarray,
        torso_height: float,
    ) -> List[np.ndarray]:
        """Lower the extracted grasp link while retaining its shelf orientation."""
        if self.retreat_distance <= 0.0:
            raise ValueError('retreat_distance must be positive')
        previous = np.asarray(start_solution, dtype=float)
        start_position = self.chain.forward(previous)[:3, 3]
        lowered_position = start_position.copy()
        lowered_position[2] -= self.retreat_distance
        solutions: List[np.ndarray] = []
        for position in self._interpolate_positions(
            start_position,
            lowered_position,
            self.cartesian_step,
        ):
            solution, _ = self.chain.solve(
                pose_matrix(position, rotation),
                [previous],
                position_tolerance=self.position_tolerance,
                orientation_tolerance=self.orientation_tolerance,
                max_iterations=180,
                fixed_positions={'torso_lift_joint': torso_height},
            )
            if solution is None:
                rounded = np.round(np.asarray(position, dtype=float), 3)
                raise RuntimeError(
                    f'No fixed-orientation carried lowering path to {rounded}'
                )
            if float(np.max(np.abs(solution[1:] - previous[1:]))) > (
                self.cartesian_joint_step
            ):
                raise RuntimeError('Carried lowering exceeds the joint-step limit')
            solutions.append(solution)
            previous = solution
        return solutions

    def _solve_carried_cradle(
        self,
        start_solution: Sequence[float],
    ) -> np.ndarray:
        """Roll q7 after retreat and clearance extension to load the lower finger."""
        previous = np.asarray(start_solution, dtype=float)
        # arm_left_7_joint is the local grasp-axis wrist roll.  Changing only
        # this joint leaves arm links 1--6 and the grasp origin stationary,
        # unlike a full-pose IK roll.  The grasp is first carried away from the
        # shelf and arm link 5 while still vertical; only then is this local
        # roll allowed to change from bilateral pinch to lower-jaw support.
        solution = previous.copy()
        solution[-1] += self.carried_cradle_roll
        joint_margin = 0.01
        if not (
            self.chain.lower[-1] + joint_margin
            <= solution[-1]
            <= self.chain.upper[-1] - joint_margin
        ):
            raise RuntimeError('Carried cradle wrist roll exceeds joint limits')
        achieved_retracted = self.chain.forward(previous)
        achieved = self.chain.forward(solution)
        expected = pose_matrix(
            achieved_retracted[:3, 3],
            achieved_retracted[:3, :3] @ _rotation_x(self.carried_cradle_roll),
        )
        cradle_error = self.chain.pose_error(achieved, expected)
        if (
            np.linalg.norm(cradle_error[:3]) > self.position_tolerance
            or np.linalg.norm(cradle_error[3:]) > self.orientation_tolerance
        ):
            raise RuntimeError('arm_left_7 is not a pure local-axis cradle roll')
        if float(achieved[2, 1]) < (
            self.carried_supported_jaw_vertical_component
        ):
            raise RuntimeError('Carried cradle does not load the supporting finger')
        return solution

    def _solve_post_retreat_clearance_extension(
        self,
        start_solution: Sequence[float],
        *, extension_distance: Optional[float] = None,
    ) -> List[np.ndarray]:
        """Move a vertical pinch forward before the post-retreat cradle roll."""
        previous = np.asarray(start_solution, dtype=float)
        start_pose = self.chain.forward(previous)
        rotation = start_pose[:3, :3]
        start_position = start_pose[:3, 3]
        distance = (self.carried_cradle_extension if extension_distance is None
                    else float(extension_distance))
        if extension_distance is not None and (
                not math.isfinite(distance) or not 0. < distance <= .18):
            raise ValueError('post-retreat extension must be within (0, .18] metres')
        extended_position = start_position + np.asarray([distance, 0.0, 0.0])
        solutions: List[np.ndarray] = []
        for position in self._interpolate_positions(
            start_position,
            extended_position,
            self.cartesian_step,
        ):
            solution, _ = self.chain.solve(
                pose_matrix(position, rotation),
                [previous],
                position_tolerance=self.position_tolerance,
                orientation_tolerance=self.orientation_tolerance,
                max_iterations=220,
                fixed_positions={'torso_lift_joint': float(previous[0])},
            )
            if solution is None:
                rounded = np.round(np.asarray(position, dtype=float), 3)
                raise RuntimeError(
                    f'No post-retreat clearance extension path to {rounded}'
                )
            if float(np.max(np.abs(solution[1:] - previous[1:]))) > (
                self.cartesian_joint_step
            ):
                raise RuntimeError(
                    'Post-retreat clearance extension exceeds the joint-step limit'
                )
            solutions.append(np.asarray(solution, dtype=float))
            previous = np.asarray(solution, dtype=float)
        return solutions

    def _solve_supported_post_retreat_staging(
        self,
        start_solution: Sequence[float],
        *, vertical_offset: Optional[float] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Reach the audited low cradle only after the shelf-clear retreat.

        ``start_solution`` has already been extended away from arm link 5 and
        rolled into gravity support.  Lower in place, then retract only the
        configured short distance.  Keeping most of the clearance extension
        through this staging avoids sweeping the book back across the arm.
        """
        first = np.asarray(start_solution, dtype=float)
        start_pose = self.chain.forward(first)
        rotation = start_pose[:3, :3]
        start_position = start_pose[:3, 3]
        offset = (-self.retreat_distance if vertical_offset is None
                  else float(vertical_offset))
        if vertical_offset is not None and (
                not math.isfinite(offset) or not -.18 <= offset <= .30 or offset == 0.):
            raise ValueError('supported staging offset must be nonzero within [-.18, .30] metres')
        lowered_position = start_position + np.asarray([0.0, 0.0, offset])
        retracted_position = lowered_position.copy()
        retracted_position[0] = (
            start_position[0] - self.carried_cradle_retraction
        )

        def solve_segment(
            previous: np.ndarray,
            segment_start: np.ndarray,
            segment_end: np.ndarray,
            label: str,
        ) -> List[np.ndarray]:
            solutions: List[np.ndarray] = []
            for position in self._interpolate_positions(
                segment_start,
                segment_end,
                self.cartesian_step,
            ):
                solution, _ = self.chain.solve(
                    pose_matrix(position, rotation),
                    [previous],
                    position_tolerance=self.position_tolerance,
                    orientation_tolerance=self.orientation_tolerance,
                    max_iterations=220,
                    fixed_positions={'torso_lift_joint': float(first[0])},
                )
                if solution is None:
                    rounded = np.round(np.asarray(position, dtype=float), 3)
                    raise RuntimeError(
                        f'No supported post-retreat {label} path to {rounded}'
                    )
                if float(np.max(np.abs(solution[1:] - previous[1:]))) > (
                    self.cartesian_joint_step
                ):
                    raise RuntimeError(
                        f'Supported post-retreat {label} exceeds the joint-step limit'
                    )
                solutions.append(np.asarray(solution, dtype=float))
                previous = np.asarray(solution, dtype=float)
            return solutions

        lowering = solve_segment(
            first,
            start_position,
            lowered_position,
            'lowering',
        )
        retraction = solve_segment(
            lowering[-1],
            lowered_position,
            retracted_position,
            'retraction',
        )
        # Preserve the historical three-section return shape.  Extension is
        # now deliberately solved and executed before the q7 roll.
        return [], lowering, retraction

    def _plan_shelf_side_cradle(
        self,
        front: Sequence[float],
        grasp_solution: Sequence[float],
        clearance_solution: Sequence[float],
        attached_corners: np.ndarray,
    ) -> List[np.ndarray]:
        """Preflight lower-jaw support before base motion, then a cached tuck.

        This development mode uses the measured grasp, not a seed-specific
        joint trajectory. It must pass the actual tool meshes as well as the
        ordinary robot/payload sweep before any arm command is sent.
        """
        from .shelf_cradle_geometry import (
            check_cradle_tool_route, check_cradle_tool_sweep,
        )

        start = np.asarray(clearance_solution, dtype=float)
        # In the official arena the shelf lip projects 65 mm in front of the
        # resting book spine. Using the spine itself would overstate clearance.
        shelf_plane = float(np.asarray(front, dtype=float)[0]) - 0.065
        cradle = self._solve_carried_cradle(start)
        route = self._plan_carried_joint_route(
            start,
            [cradle],
            attached_corners,
            post_retreat_shelf_front_x=shelf_plane,
        )
        if not route:
            raise RuntimeError('No payload-safe shelf-side cradle route')
        reason = check_cradle_tool_sweep(
            self, front, grasp_solution, start, cradle, shelf_plane,
            aperture=float(self.carried_book_dimensions[1]),
        )
        if reason is not None:
            raise RuntimeError(f'Shelf-side cradle tool sweep rejected: {reason}')

        post_retreat_plane = shelf_plane + self.carried_shelf_retreat_clearance
        _, lowering, retraction = self._solve_supported_post_retreat_staging(cradle)
        previous = np.asarray(cradle, dtype=float)
        cached_legs: List[Tuple[np.ndarray, str]] = []
        for goals, phase in (
            (lowering, 'supported_cradle_lowering'),
            (retraction, 'supported_cradle_retraction'),
        ):
            section = self._plan_carried_joint_route(
                previous, goals, attached_corners,
                post_retreat_shelf_front_x=post_retreat_plane,
                require_gravity_support=True,
            )
            if not section:
                raise RuntimeError(f'No payload-safe {phase} route')
            cached_legs.extend((np.asarray(q).copy(), phase) for q in section)
            previous = np.asarray(section[-1], dtype=float)
        staging_terminal = previous.copy()
        compact = self._plan_carried_joint_route(
            previous, self._supported_compact_goals(previous), attached_corners,
            post_retreat_shelf_front_x=post_retreat_plane,
            require_gravity_support=True,
        )
        if not compact:
            raise RuntimeError('No payload-safe supported compact route')
        radius = self._carried_navigation_radius(compact[-1], attached_corners)
        if radius > self.carried_navigation_radius_limit:
            raise RuntimeError('Supported compact route exceeds navigation radius')
        cached_legs.extend((np.asarray(q).copy(), 'compact_transport') for q in compact)
        reason = check_cradle_tool_route(
            self, front, grasp_solution, cradle,
            [q for q, _ in cached_legs], post_retreat_plane,
            aperture=float(self.carried_book_dimensions[1]),
        )
        if reason is not None:
            raise RuntimeError(f'Supported compact tool sweep rejected: {reason}')
        self._cached_post_retreat_plan = {
            'start': np.asarray(cradle).copy(),
            'shelf_front_x': post_retreat_plane,
            'attached_corners': attached_corners.copy(),
            'requires_gravity_support': True,
            'legs': cached_legs,
            'staging_terminal': staging_terminal,
            'terminal': np.asarray(compact[-1]).copy(),
            'compact_radius': float(radius),
        }
        # The checked q7 sweep is dispatched continuously. Splitting it into
        # three endpoint waits would pause while the support mode is changing.
        return [np.asarray(cradle).copy()]

    def _plan_carried_return(
        self,
        front: Sequence[float],
        grasp_solution: Sequence[float],
        clearance_solution: Sequence[float],
        rotation: np.ndarray,
        torso_height: float,
        *, geometry_backend=None, defer_return: bool = False,
        extension_distance: Optional[float] = None,
        staging_vertical_offset: Optional[float] = None,
        compact_path: str = 'standard',
    ) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        Optional[int],
        List[np.ndarray],
        np.ndarray,
    ]:
        """Plan a payload-safe shelf exit and any deferred compact staging."""
        self._cached_post_retreat_plan = None
        attached_corners = self._attached_book_corners(front, grasp_solution)
        if compact_path not in ('standard', 'middle', 'bottom'):
            raise ValueError('unknown supported compact path')
        if not defer_return and (extension_distance is not None
                or staging_vertical_offset is not None or compact_path != 'standard'):
            raise ValueError('lower shelf staging options require a deferred return')
        if type(defer_return) is not bool:
            raise ValueError('defer_return must be boolean')
        requires_post_retreat_return = (
            defer_return or float(np.asarray(front, dtype=float)[2]) >= 1.42
        )
        if geometry_backend is not None and (not requires_post_retreat_return
                or getattr(self, 'shelf_side_cradle_enabled', False)):
            raise ValueError('parallel carry requires the ordinary post-retreat branch')
        geometry_options = {} if geometry_backend is None else {'geometry_backend': geometry_backend}
        shelf_front_x = float(np.asarray(front, dtype=float)[0])
        if requires_post_retreat_return:
            if getattr(self, 'shelf_side_cradle_enabled', False):
                cradle = self._plan_shelf_side_cradle(
                    front, grasp_solution, clearance_solution, attached_corners
                )
                return [], cradle, 0, [], attached_corners
            vertical_start = np.asarray(clearance_solution, dtype=float)
            # Keep the proven bilateral vertical pinch throughout the fixed-
            # heading shelf retreat.  Once the base is clear, translate the
            # grasp origin away from arm link 5 before changing its attitude.
            post_retreat_front_x = (
                shelf_front_x + self.carried_shelf_retreat_clearance
            )
            extension_goals = self._solve_post_retreat_clearance_extension(
                vertical_start,
                **({'extension_distance': extension_distance}
                   if extension_distance is not None else {}),
            )
            validated_extension = self._plan_carried_joint_route(
                vertical_start,
                extension_goals,
                attached_corners,
                post_retreat_shelf_front_x=post_retreat_front_x,
                **geometry_options,
            )
            if not validated_extension:
                raise RuntimeError('No payload-safe post-retreat clearance extension')
            extended_solution = np.asarray(
                validated_extension[-1], dtype=float
            )

            # The q7-only local roll is subdivided for dense collision checks
            # but is dispatched as one continuous controller goal.  This mode
            # transition is intentionally deferred until after both retreat
            # and extension: live shelf-side rolls slipped onto arm_left_5.
            cradle_solution = self._solve_carried_cradle(extended_solution)
            validated_cradle = self._plan_carried_joint_route(
                extended_solution,
                [cradle_solution],
                attached_corners,
                post_retreat_shelf_front_x=post_retreat_front_x,
                **geometry_options,
            )
            if validated_cradle is None:
                raise RuntimeError('No payload-safe mechanical cradle route')

            # Preflight the exact supported lowering and compaction while the
            # measured target pose and attached envelope remain available.
            _, supported_lowering, retraction = (
                self._solve_supported_post_retreat_staging(cradle_solution,
                    **({'vertical_offset': staging_vertical_offset}
                       if staging_vertical_offset is not None else {}))
            )
            staged_previous = np.asarray(cradle_solution, dtype=float)
            cached_legs: List[Tuple[np.ndarray, str]] = [
                *(
                    (
                        np.asarray(solution, dtype=float).copy(),
                        'post_retreat_clearance_extension',
                    )
                    for solution in validated_extension
                ),
                *(
                    (
                        np.asarray(solution, dtype=float).copy(),
                        'post_retreat_cradle_roll',
                    )
                    for solution in validated_cradle
                ),
            ]
            for goals, phase, failure in (
                (
                    supported_lowering,
                    'supported_cradle_lowering',
                    'lowering',
                ),
                (
                    retraction,
                    'supported_cradle_retraction',
                    'retraction',
                ),
            ):
                planned_section = self._plan_carried_joint_route(
                    staged_previous,
                    goals,
                    attached_corners,
                    post_retreat_shelf_front_x=post_retreat_front_x,
                    **geometry_options,
                    require_gravity_support=True,
                )
                if planned_section is None:
                    raise RuntimeError(
                        f'No payload-safe supported post-retreat {failure} route'
                    )
                cached_legs.extend(
                    (
                        np.asarray(solution, dtype=float).copy(),
                        phase,
                    )
                    for solution in planned_section
                )
                if planned_section:
                    staged_previous = np.asarray(
                        planned_section[-1], dtype=float
                    )

            staging_terminal = staged_previous.copy()
            compact_route = None
            # Keep the existing synchronized tuck first.  Two bounded
            # shoulder leads address a shelf-intersecting intermediate sweep
            # without changing the terminal posture or collision margins.
            if compact_path in ('middle', 'bottom'):
                from .lower_shelf_carry import middle_compact_proposal, bottom_compact_proposal
                proposal = (middle_compact_proposal if compact_path == 'middle'
                            else bottom_compact_proposal)
                compact_goals = proposal(self, staged_previous)
                compact_route = self._plan_carried_joint_route(
                    staged_previous, compact_goals, attached_corners,
                    post_retreat_shelf_front_x=post_retreat_front_x,
                    **geometry_options, require_gravity_support=True,
                )
                shoulder_progress_power = None
            else:
                for shoulder_progress_power in (1.0, 0.8, 0.65):
                    try:
                        compact_goals = (
                            self._supported_compact_goals(staged_previous)
                            if shoulder_progress_power == 1.0
                            else self._supported_compact_goals(
                                staged_previous,
                                shoulder_progress_power=shoulder_progress_power,
                            )
                        )
                    except RuntimeError:
                        continue
                    compact_route = self._plan_carried_joint_route(
                        staged_previous,
                        compact_goals,
                        attached_corners,
                        post_retreat_shelf_front_x=post_retreat_front_x,
                        **geometry_options,
                        require_gravity_support=True,
                    )
                    if compact_route is not None:
                        break
            if compact_route is None:
                raise RuntimeError('No payload-safe supported compact transport route')
            compact_radius = self._carried_navigation_radius(
                compact_route[-1],
                attached_corners,
            )
            if compact_radius > self.carried_navigation_radius_limit:
                raise RuntimeError('No payload-safe compact transport route')
            cached_legs.extend(
                (
                    np.asarray(solution, dtype=float).copy(),
                    'compact_transport',
                )
                for solution in compact_route
            )
            cached_plan = {
                'start': vertical_start.copy(),
                'shelf_front_x': post_retreat_front_x,
                'attached_corners': attached_corners.copy(),
                'requires_gravity_support': False,
                'legs': cached_legs,
                'staging_terminal': staging_terminal,
                'terminal': np.asarray(compact_route[-1], dtype=float).copy(),
                'compact_radius': float(compact_radius),
                'compact_shoulder_progress_power': shoulder_progress_power,
            }
            if defer_return:
                from .shelf_cradle_geometry import check_cradle_tool_route
                tool_route = (check_cradle_tool_route if geometry_backend is None
                              else geometry_backend.tool_route)
                tool_args = ((self,) if geometry_backend is None else ())
                reason = tool_route(
                    *tool_args, front, grasp_solution, vertical_start,
                    [q for q, _ in cached_legs], post_retreat_front_x,
                    aperture=float(self.carried_book_dimensions[1]),
                )
                if reason is not None:
                    raise RuntimeError(f'Lower shelf carried tool route rejected: {reason}')
                cached_plan['compact_path'] = compact_path
                cached_plan['staging_vertical_offset_m'] = staging_vertical_offset
            self._cached_post_retreat_plan = cached_plan
            return (
                [],
                [],
                None,
                [],
                attached_corners,
            )

        lowering = self._solve_fixed_orientation_lowering(
            clearance_solution,
            rotation,
            torso_height,
        )
        previous = np.asarray(clearance_solution, dtype=float)
        for solution in lowering:
            if not self._carried_transition_is_safe(
                previous,
                solution,
                attached_corners,
                shelf_front_x,
            ):
                raise RuntimeError('Inflated book sweep is unsafe during carried lowering')
            previous = solution

        home_at_height = HOME.copy()
        home_at_height[0] = torso_height
        staging = PREGRASP.copy()
        staging[0] = torso_height
        route_candidates = [[home_at_height], [staging, home_at_height]]
        # A synchronized interpolation of every joint can swing the payload
        # toward the shelf even when both endpoint poses are clear.  Also try
        # reaching HOME while delaying one joint, then finish that single
        # joint in a second checked leg.  This small family captures a safe
        # tuck without hard-coding which joint must be delayed for a given IK
        # branch or measured book pose.
        for joint_index in range(1, len(home_at_height)):
            delayed_joint = home_at_height.copy()
            delayed_joint[joint_index] = previous[joint_index]
            route_candidates.append([delayed_joint, home_at_height])
        best_route = None
        for route in route_candidates:
            expanded_route = self._plan_carried_joint_route(
                previous,
                route,
                attached_corners,
                shelf_front_x=shelf_front_x,
            )
            if expanded_route is None:
                continue
            route_previous = expanded_route[-1]
            route_cost = self._arm_route_cost(
                previous,
                expanded_route[:-1],
                route_previous,
            )
            transport_pose = np.asarray(route_previous, dtype=float).copy()
            transport_pose[0] = HOME[0]
            if not self._carried_transition_is_safe(
                route_previous,
                transport_pose,
                attached_corners,
                shelf_front_x,
            ):
                continue
            if best_route is None or route_cost < best_route[0]:
                best_route = (route_cost, expanded_route)
        if best_route is None:
            transport_pose = previous.copy()
            transport_pose[0] = HOME[0]
            if not self._carried_transition_is_safe(
                previous,
                transport_pose,
                attached_corners,
                shelf_front_x,
            ):
                raise RuntimeError('No payload-safe carried route from shelf')
            compact_route = self._plan_carried_joint_route(
                transport_pose,
                [CARRY],
                attached_corners,
                post_retreat_shelf_front_x=(
                    shelf_front_x + self.carried_shelf_retreat_clearance
                ),
            )
            if compact_route is None or self._carried_navigation_radius(
                compact_route[-1],
                attached_corners,
            ) > self.carried_navigation_radius_limit:
                raise RuntimeError('No payload-safe compact transport route')
            return (
                lowering,
                [],
                None,
                [],
                attached_corners,
            )
        return lowering, [], None, best_route[1], attached_corners

    @staticmethod
    def _subset_arm_state(
        start: Sequence[float],
        goal: Sequence[float],
        mask: int,
    ) -> np.ndarray:
        state = np.asarray(start, dtype=float).copy()
        target = np.asarray(goal, dtype=float)
        for arm_index in range(len(ARM_JOINTS)):
            if mask & (1 << arm_index):
                state[arm_index + 1] = target[arm_index + 1]
        return state

    def _plan_retracted_transition(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        *,
        edge_validator: Optional[Callable[[Sequence[float], Sequence[float]], bool]] = None,
    ) -> Optional[List[np.ndarray]]:
        """Return checked intermediate waypoints, excluding the final goal.

        Direct and legacy PREGRASP transitions remain preferred.  If both
        sweep outside the retracted envelope, a deterministic breadth-first
        search considers the 2^7 states formed by taking each arm joint from
        either endpoint.  The validated top-row branch follows joints 1..7.
        """
        def edge_is_safe(first_state, last_state):
            if edge_validator is not None and not edge_validator(first_state, last_state):
                return False
            return self._retracted_transition_is_safe(first_state, last_state)

        first = np.asarray(start, dtype=float)
        last = np.asarray(goal, dtype=float)
        if edge_is_safe(first, last):
            return []

        staging = PREGRASP.copy()
        staging[0] = last[0]
        if edge_is_safe(
            first, staging
        ) and edge_is_safe(staging, last):
            return [staging]

        goal_mask = (1 << len(ARM_JOINTS)) - 1
        parents: Dict[int, Optional[int]] = {0: None}
        frontier = deque([0])
        while frontier:
            mask = frontier.popleft()
            state = self._subset_arm_state(first, last, mask)
            if mask == goal_mask:
                break
            for arm_index in range(len(ARM_JOINTS)):
                bit = 1 << arm_index
                if mask & bit:
                    continue
                next_mask = mask | bit
                if next_mask in parents:
                    continue
                next_state = self._subset_arm_state(first, last, next_mask)
                if not edge_is_safe(state, next_state):
                    continue
                parents[next_mask] = mask
                frontier.append(next_mask)

        if goal_mask not in parents:
            return None
        masks = []
        cursor = goal_mask
        while cursor:
            masks.append(cursor)
            parent = parents[cursor]
            if parent is None:
                break
            cursor = parent
        masks.reverse()
        # The Cartesian solution already contains the final clearance pose.
        return [
            self._subset_arm_state(first, last, mask)
            for mask in masks[:-1]
        ]

    @staticmethod
    def _arm_route_cost(
        start: Sequence[float],
        intermediates: Sequence[Sequence[float]],
        goal: Sequence[float],
    ) -> float:
        points = [
            np.asarray(start, dtype=float),
            *(np.asarray(q, dtype=float) for q in intermediates),
            np.asarray(goal, dtype=float),
        ]
        return sum(
            float(np.linalg.norm(second[1:] - first[1:]))
            for first, second in zip(points, points[1:])
        )

    def _solve_cartesian_path(
        self,
        positions: Sequence[Sequence[float]],
        rotations: Sequence[np.ndarray],
        torso_height: float,
        *,
        endpoint_first: bool = False,
        transition_start: Optional[Sequence[float]] = None,
        candidate_validator: Optional[
            Callable[[Sequence[np.ndarray], Sequence[np.ndarray]], bool]
        ] = None,
        first_valid: bool = False,
        joint_limit_margin: float = 0.01,
        skip_setup_transition: bool = False,
        transition_edge_validator: Optional[Callable[[Sequence[float], Sequence[float]], bool]] = None,
        setup_transition_planner: Optional[
            Callable[[Sequence[float], Sequence[float]], Optional[List[np.ndarray]]]
        ] = None,
        position_tolerance: Optional[float] = None,
        orientation_tolerance: Optional[float] = None,
    ) -> Tuple[List[np.ndarray], int, float, List[np.ndarray]]:
        if not positions or not rotations:
            raise ValueError('Cartesian path requires positions and rotations')
        if setup_transition_planner is not None and (
            skip_setup_transition or transition_edge_validator is None or candidate_validator is None
        ):
            raise ValueError('Custom setup planner requires edge and full candidate validation')
        solve_position_tolerance = (
            self.position_tolerance if position_tolerance is None else float(position_tolerance)
        )
        solve_orientation_tolerance = (
            self.orientation_tolerance if orientation_tolerance is None else float(orientation_tolerance)
        )
        if not (
            np.isfinite(solve_position_tolerance) and solve_position_tolerance > 0.0
            and np.isfinite(solve_orientation_tolerance) and solve_orientation_tolerance > 0.0
        ):
            raise ValueError('Cartesian IK tolerances must be finite and positive')
        current = np.asarray(
            self._current_seed() if transition_start is None else transition_start,
            dtype=float,
        ).copy()
        current[0] = torso_height
        seeds = []
        # The top-row branch was mesh-audited from the OFFER elbow posture.
        # Solving its grasp endpoint from another seed can converge to a
        # numerically valid wrist-below-shelf branch that the link-origin
        # collision checks cannot see, so keep this selection deterministic.
        seed_templates = (OFFER,) if endpoint_first else (
            current,
            HOME,
            OFFER,
            PREGRASP,
        )
        for seed in seed_templates:
            candidate = np.asarray(seed, dtype=float).copy()
            candidate[0] = torso_height
            seeds.append(candidate)

        best = None
        for orientation_index, rotation in enumerate(rotations):
            for seed in seeds:
                ordered_positions = (
                    list(reversed(positions)) if endpoint_first else list(positions)
                )
                solution, first_score = self.chain.solve(
                    pose_matrix(ordered_positions[0], rotation),
                    [seed],
                    position_tolerance=solve_position_tolerance,
                    orientation_tolerance=solve_orientation_tolerance,
                    max_iterations=180,
                    fixed_positions={'torso_lift_joint': torso_height},
                    joint_limit_margin=joint_limit_margin,
                )
                if solution is None:
                    continue

                ordered_solutions: List[np.ndarray] = [solution]
                scores: List[float] = [float(first_score)]
                previous = solution
                for position in ordered_positions[1:]:
                    solution, score = self.chain.solve(
                        pose_matrix(position, rotation),
                        [previous],
                        position_tolerance=solve_position_tolerance,
                        orientation_tolerance=solve_orientation_tolerance,
                        max_iterations=180,
                        fixed_positions={'torso_lift_joint': torso_height},
                        joint_limit_margin=joint_limit_margin,
                    )
                    if solution is None:
                        break
                    if float(np.max(np.abs(solution[1:] - previous[1:]))) > (
                        self.cartesian_joint_step
                    ):
                        break
                    ordered_solutions.append(solution)
                    scores.append(float(score))
                    previous = solution
                if len(ordered_solutions) != len(positions):
                    continue

                solutions = (
                    list(reversed(ordered_solutions))
                    if endpoint_first
                    else ordered_solutions
                )
                # The preparation command independently checks the full empty
                # setup at a farther base pose, including fingers and torso.
                if skip_setup_transition:
                    transition = []
                elif setup_transition_planner is not None:
                    transition = setup_transition_planner(current, solutions[0])
                else:
                    transition = self._plan_retracted_transition(
                        current, solutions[0],
                        **({'edge_validator': transition_edge_validator}
                           if transition_edge_validator is not None else {}),
                    )
                if transition is None:
                    continue
                transition_cost = self._arm_route_cost(
                    current,
                    transition,
                    solutions[0],
                )
                joint_cost = sum(
                    float(np.linalg.norm(second[1:] - first[1:]))
                    for first, second in zip(solutions, solutions[1:])
                )
                candidate_score = transition_cost + joint_cost + sum(scores)
                if best is not None and candidate_score >= best[0]:
                    continue
                if candidate_validator is not None and not candidate_validator(
                    solutions,
                    transition,
                ):
                    continue
                if first_valid:
                    return (
                        solutions,
                        orientation_index,
                        candidate_score,
                        transition,
                    )
                best = (
                    candidate_score,
                    orientation_index,
                    solutions,
                    transition,
                )

        if best is None:
            rounded = np.round(np.asarray(positions[-1], dtype=float), 3)
            raise RuntimeError(f'No continuous collision-aware IK path to {rounded}')
        score, orientation_index, solutions, transition = best
        return solutions, orientation_index, score, transition

    def _move_torso(self, height: float, duration: float = 2.0, *, planned_start=None,
                    allow_completed_hold=False, empty_torso_owner=None) -> bool:
        if allow_completed_hold and getattr(self, 'settled_place_torso_skip_enabled', False):
            held = try_completed_torso_hold(self, height,
                require_contact=lambda: _require_place_contact_clear(self),
                check_scene=lambda reference: measured_scene_context(self, reference))
            if held is not None:
                return held
        return self._follow(
            self.torso_client,
            ['torso_lift_joint'],
            [height],
            duration,
            **({'empty_torso_owner': empty_torso_owner} if empty_torso_owner is not None else {}),
            **({'pre_send_check': lambda: require_planned_place_height(self, planned_start)}
               if planned_start is not None else {}),
        )

    def _move_arm_solution(self, solution: Sequence[float], duration: float, *, trajectory_duration=None) -> bool:
        q = np.asarray(solution, dtype=float)
        return self._follow(self.arm_client, ARM_JOINTS, q[1:], duration,
            **({'trajectory_duration': trajectory_duration} if trajectory_duration is not None else {}))

    @staticmethod
    def _transport_leg_duration(
        start: Sequence[float],
        end: Sequence[float],
    ) -> float:
        """Scale subdivided transport legs without making tiny moves 2.8 s long."""
        first = np.asarray(start, dtype=float)
        last = np.asarray(end, dtype=float)
        maximum_delta = float(np.max(np.abs(last[1:] - first[1:])))
        return float(np.clip(4.0 * maximum_delta, 0.35, 2.8))

    def _make_retained_arm_trajectory_goal(
        self,
        legs: Sequence[Tuple[Sequence[float], float, str]],
    ) -> Tuple[FollowJointTrajectory.Goal, float]:
        """Build one continuous controller goal from prevalidated carried legs."""
        if not legs:
            raise ValueError('retained arm trajectory requires at least one leg')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(ARM_JOINTS)
        elapsed = 0.0
        for solution, duration, _ in legs:
            q = np.asarray(solution, dtype=float)
            seconds = float(duration)
            if (
                q.shape != (8,)
                or not np.all(np.isfinite(q))
                or not np.isfinite(seconds)
                or seconds <= 0.0
            ):
                raise ValueError('retained arm trajectory contains an invalid leg')
            elapsed += seconds
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in q[1:]]
            point.time_from_start = Duration(seconds=elapsed).to_msg()
            goal.trajectory.points.append(point)
        return goal, elapsed

    @staticmethod
    def _trajectory_leg_at_elapsed_time(
        legs: Sequence[Tuple[Sequence[float], float, str]],
        elapsed: float,
    ) -> Tuple[int, str]:
        cumulative = 0.0
        for index, (_, duration, phase) in enumerate(legs):
            cumulative += float(duration)
            if elapsed <= cumulative:
                return index, phase
        return len(legs) - 1, legs[-1][2]

    @staticmethod
    def _valid_retained_terminal_result(result) -> bool:
        status = getattr(result, 'status', None)
        return bool(
            isinstance(status, int) and not isinstance(status, bool)
            and status in (
                GoalStatus.STATUS_SUCCEEDED,
                GoalStatus.STATUS_CANCELED,
                GoalStatus.STATUS_ABORTED,
            )
        )

    def _cancel_retained_goal_and_confirm(self, handle, result_future) -> bool:
        if not self._cancel_goal_and_confirm(handle, result_future):
            return False
        try:
            return self._valid_retained_terminal_result(result_future.result())
        except Exception:
            return False

    def _cancel_late_retained_goal(self, acceptance_future, token) -> None:
        """Cancel a goal accepted after its bounded wait failed; never recover."""
        try:
            handle = acceptance_future.result()
            if handle is None or not isinstance(handle.accepted, bool):
                return
            if not handle.accepted:
                with self._lock:
                    self._pending_retained_acceptances.discard(token)
                return
            with self._lock:
                if handle not in self._goal_handles:
                    self._goal_handles.append(handle)
            # Request the stop even if obtaining its result future then fails.
            try:
                handle.cancel_goal_async()
            except Exception:
                pass
            terminal = handle.get_result_async()

            def forget_terminal(future) -> None:
                try:
                    confirmed = future.done() and self._valid_retained_terminal_result(
                        future.result())
                except Exception:
                    confirmed = False
                if confirmed:
                    with self._lock:
                        if handle in self._goal_handles:
                            self._goal_handles.remove(handle)
                        self._pending_retained_acceptances.discard(token)

            terminal.add_done_callback(forget_terminal)
        except Exception:
            # Keep any accepted handle registered for shutdown. The original
            # acceptance failure already set cancellation and raised a terminal
            # error; a callback failure must not enable recovery or new goals.
            return

    def _send_retained_arm_trajectory(
        self,
        goal: FollowJointTrajectory.Goal,
        total_duration: float,
        legs: Sequence[Tuple[Sequence[float], float, str]],
        command: str,
        *,
        leg_offset: int = 0,
        initial_pressure_gate=None,
        velocity_admission: bool = False,
        velocity_headroom: bool = False,
        clearance_admission=None,
        release_clearance_admission=None,
        withdrawal_admission=None,
        transition_stop=None,
    ) -> Tuple[bool, bool]:
        """Send a continuous carried route and cancel it on payload danger."""
        if self._cancel.is_set():
            return False, False
        if type(velocity_admission) is not bool:
            raise ArmVelocityAdmissionRejected('faster retained admission flag must be Boolean')
        if type(velocity_headroom) is not bool or (velocity_headroom and not velocity_admission):
            raise ArmVelocityAdmissionRejected('retained headroom requires faster admission')
        if clearance_admission is not None and not (velocity_admission and velocity_headroom):
            raise ArmVelocityAdmissionRejected('clearance timing requires retained headroom admission')
        if withdrawal_admission is not None and (not (velocity_admission and velocity_headroom)
                or clearance_admission is not None):
            raise ArmVelocityAdmissionRejected('withdrawal timing requires exclusive retained headroom admission')
        if release_clearance_admission is not None and (
                not (velocity_admission and velocity_headroom)
                or clearance_admission is not None or withdrawal_admission is not None
                or transition_stop is not None or total_duration != 2.8):
            raise ArmVelocityAdmissionRejected('release-only clearance requires exclusive retained headroom')
        if velocity_admission and initial_pressure_gate is not None:
            raise ArmVelocityAdmissionRejected('faster retained arm cannot use the lift pressure gate')

        endpoint_contact_age = min(
            0.15,
            float(getattr(self, 'grasp_contact_max_age', 0.15)),
        )

        def watchdog_fault() -> Optional[str]:
            return self._payload_hazard_reason(max_age=endpoint_contact_age)

        def publish_fault(reason: str, started_ns: int) -> None:
            now_ns = self.get_clock().now().nanoseconds
            elapsed = max(0.0, (now_ns - started_ns) / 1e9)
            leg, phase = self._trajectory_leg_at_elapsed_time(legs, elapsed)
            self._publish_status(
                'grasp_lost',
                command=command,
                phase=phase,
                leg=leg + leg_offset,
                reason=reason,
                **(_place_transition_stop.diagnostic_fields(self, reason)
                   if getattr(self, 'place_transition_stop_enabled', False) else {}),
            )

        # These constant-time gates catch a collision/contact loss that arrived
        # during the final probe callback itself without adding a planning dwell.
        initial_fault = watchdog_fault()
        if initial_fault is not None:
            publish_fault(initial_fault, self.get_clock().now().nanoseconds)
            return False, True

        acceptance_future = None
        acceptance_fault = None
        acceptance_token = object()
        velocity_record = None
        velocity_rejection = None
        def request_goal():
            nonlocal velocity_record, velocity_rejection, goal, legs
            if velocity_admission:
                try:
                    velocity_record = require_arm_velocity_locked(self, goal)
                    if getattr(self, 'additional_arm_time_scale', 1.0) != 1.0:
                        goal, legs, timing_record = retime_admitted_arm_goal(
                            goal, velocity_record, self.additional_arm_time_scale,
                            nominal_duration=total_duration, legs=legs)
                        velocity_record = require_arm_velocity_locked(self, goal)
                        require_retimed_arm_headroom(velocity_record)
                        velocity_record['additional_arm_timing'] = timing_record
                        if self._cancel.is_set():
                            raise ArmVelocityAdmissionRejected('cancelled before retimed arm dispatch', velocity_record)
                    if velocity_headroom:
                        require_retimed_arm_headroom(velocity_record)
                    if clearance_admission is not None:
                        require_bin_clearance_publication_locked(self, clearance_admission, goal, command)
                    if release_clearance_admission is not None:
                        release_clearance_admission.require_publication_locked(self, goal, command, leg_offset)
                    if withdrawal_admission is not None:
                        require_withdrawal_half_publication_locked(
                            self, withdrawal_admission, goal, command, leg_offset)
                except ArmVelocityAdmissionRejected as error:
                    velocity_rejection = error
                    raise
            # The pressure gate already holds the sensor mutex at admission.
            # Ordinary callers register under that mutex immediately below.
            if transition_stop is not None:
                transition_stop.require_publication_locked(self, goal, command, leg_offset)
            pending = getattr(self, '_pending_retained_acceptances', set())
            pending.add(acceptance_token)
            self._pending_retained_acceptances = pending
            return self.arm_client.send_goal_async(goal)
        try:
            if velocity_admission:
                with self._adaptive_command_guard():
                    if self._cancel.is_set():
                        return False, False
                    final_fault = watchdog_fault()
                    if final_fault is not None:
                        publish_fault(final_fault, self.get_clock().now().nanoseconds)
                        return False, True
                    with self._lock:
                        if self._cancel.is_set():
                            return False, False
                        acceptance_future = request_goal()
            elif initial_pressure_gate is not None:
                acceptance_future = initial_pressure_gate.send(request_goal)
            elif getattr(self, '_place_contact_guard', None) is not None:
                # Match the PLACE contact callback's command -> sensor order.
                # Its committed stop must precede or prevent this publication;
                # acceptance waiting remains outside both locks.
                with self._adaptive_command_guard():
                    if self._cancel.is_set():
                        return False, False
                    final_fault = watchdog_fault()
                    if final_fault is not None:
                        publish_fault(final_fault, self.get_clock().now().nanoseconds)
                        return False, True
                    with self._lock:
                        if self._cancel.is_set():
                            return False, False
                        acceptance_future = request_goal()
            else:
                with self._lock:
                    acceptance_future = request_goal()
            acceptance_deadline = time.monotonic() + self.timeout
            while not acceptance_future.done():
                # A hazard may be transient while the action server accepts the
                # request. Remember it so a later fresh callback cannot authorize
                # an action that already lost its payload contract.
                acceptance_fault = acceptance_fault or watchdog_fault()
                if time.monotonic() >= acceptance_deadline:
                    raise TimeoutError('retained arm goal acceptance timed out')
                time.sleep(0.02)
            goal_handle = acceptance_future.result()
            if goal_handle is None or not isinstance(goal_handle.accepted, bool):
                raise RuntimeError('retained arm goal acceptance result is invalid')
        except LiftPressureRejected:
            # Rejected before publication: no recovery route or gripper-open
            # command is authorized by this unverified loaded state.
            raise
        except Exception as exc:
            if exc is velocity_rejection:
                # Only this exact before-publication exception is a clean reject.
                # A client exception after registration retains the old unknown
                # acceptance handling, even if it has this exception's class.
                raise
            if transition_stop is not None and exc is transition_stop.publication_rejection:
                # Exact added qualification failure precedes registration/publication.
                raise
            self._cancel.set()
            if acceptance_future is not None:
                # The server might still start this request. Observe/cancel a
                # late handle without blocking this callback or guessing that a
                # timeout means the robot stopped.
                try:
                    acceptance_future.add_done_callback(
                        lambda future: self._cancel_late_retained_goal(
                            future, acceptance_token))
                except Exception:
                    pass
            raise RetainedMotionNotStopped(
                'retained arm goal acceptance has no confirmed terminal state'
            ) from exc
        finally:
            if velocity_record is not None or velocity_rejection is not None:
                publish_arm_velocity_admission(self,
                    velocity_record if velocity_rejection is None else velocity_rejection.record,
                    command=command, admitted=velocity_rejection is None,
                    reason=None if velocity_rejection is None else str(velocity_rejection))
        if not goal_handle.accepted:
            with self._lock:
                self._pending_retained_acceptances.discard(acceptance_token)
            return False, False
        with self._lock:
            self._goal_handles.append(goal_handle)
            self._pending_retained_acceptances.discard(acceptance_token)
        keep_goal_registered = True
        try:
            started_ns = self.get_clock().now().nanoseconds
            result_future = goal_handle.get_result_async()
            if acceptance_fault is not None:
                publish_fault(acceptance_fault, started_ns)
                if not self._cancel_retained_goal_and_confirm(goal_handle, result_future):
                    self._cancel.set()
                    raise RetainedMotionNotStopped(
                        'unsafe accepted arm trajectory could not be cancelled'
                    )
                keep_goal_registered = False
                return False, True
            # Gazebo commonly runs below real time on the full TIAGo world.
            # Match the simulator-aware waits elsewhere in this node instead of
            # assuming one simulated trajectory second is one wall second.
            wall_deadline = (
                time.monotonic() + self.timeout + 4.0 * total_duration
            )
            while not result_future.done():
                if self._cancel.is_set():
                    if not self._cancel_retained_goal_and_confirm(
                        goal_handle,
                        result_future,
                    ):
                        keep_goal_registered = True
                        raise RetainedMotionNotStopped(
                            'retained arm trajectory cancellation was not confirmed'
                        )
                    keep_goal_registered = False
                    return False, False
                fault = watchdog_fault()
                if fault is not None:
                    publish_fault(fault, started_ns)
                    if not self._cancel_retained_goal_and_confirm(
                        goal_handle,
                        result_future,
                    ):
                        keep_goal_registered = True
                        self._cancel.set()
                        raise RetainedMotionNotStopped(
                            'unsafe retained arm trajectory could not be cancelled'
                        )
                    keep_goal_registered = False
                    return False, True
                if time.monotonic() >= wall_deadline:
                    if not self._cancel_retained_goal_and_confirm(
                        goal_handle,
                        result_future,
                    ):
                        keep_goal_registered = True
                        self._cancel.set()
                        raise RetainedMotionNotStopped(
                            'timed-out retained arm trajectory could not be cancelled'
                        )
                    keep_goal_registered = False
                    raise TimeoutError('retained arm trajectory timed out')
                time.sleep(0.02)
            wrapped = result_future.result()
            if not self._valid_retained_terminal_result(wrapped):
                raise RuntimeError('retained arm result has no valid terminal status')
            if self._cancel.is_set():
                # Cancellation can race an already terminal result, bypassing
                # the polling loop. A stopped action must not advance the route.
                keep_goal_registered = False
                return False, False
            if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
                keep_goal_registered = False
                return False, False
            fault = watchdog_fault()
            if fault is not None:
                publish_fault(fault, started_ns)
                keep_goal_registered = False
                return False, True
            keep_goal_registered = False
            return True, False
        except RetainedMotionNotStopped:
            raise
        except Exception as exc:
            if keep_goal_registered:
                self._cancel.set()
                try:
                    goal_handle.cancel_goal_async()
                except Exception:
                    pass
                raise RetainedMotionNotStopped(
                    'accepted retained arm action has no confirmed terminal state'
                ) from exc
            raise
        finally:
            if not keep_goal_registered:
                with self._lock:
                    if goal_handle in self._goal_handles:
                        self._goal_handles.remove(goal_handle)

    def _wait_for_retained_endpoint(
        self,
        expected: Sequence[float],
        *,
        command: str,
        phase: str,
        leg: Optional[int] = None,
        settle_timeout: float = 2.0,
        torso_tolerance: float = 0.001,
        arm_tolerance: float = 0.005,
    ) -> Optional[np.ndarray]:
        """Wait for Gazebo's measured joints to catch a completed action goal.

        The trajectory controller can report success one joint-state callback
        before the simulated position interface reaches its final command.  Do
        not mistake that short residual motion for an endpoint miss.  The wait
        remains fail-closed and continuously checks the held payload whenever
        convergence is not already visible in the first measurement.
        """
        target = np.asarray(expected, dtype=float)
        if target.shape != (8,) or not np.all(np.isfinite(target)):
            raise ValueError('retained endpoint target is invalid')
        if (
            not np.isfinite(settle_timeout)
            or settle_timeout < 0.0
            or not np.isfinite(torso_tolerance)
            or torso_tolerance < 0.0
            or not np.isfinite(arm_tolerance)
            or arm_tolerance < 0.0
        ):
            raise ValueError('retained endpoint wait parameters are invalid')

        endpoint_contact_age = min(
            0.15,
            float(getattr(self, 'grasp_contact_max_age', 0.15)),
        )
        started_ns: Optional[int] = None
        required_ns = int(settle_timeout * 1e9)
        wall_deadline: Optional[float] = None
        last_torso_error = float('inf')
        last_arm_error = float('inf')

        def publish_grasp_fault(reason: str) -> None:
            fields = {
                'command': command,
                'phase': phase,
                'reason': reason,
                'endpoint_settle': True,
            }
            if leg is not None:
                fields['leg'] = leg
            self._publish_status('grasp_lost', **fields)

        while True:
            cancel = getattr(self, '_cancel', None)
            if cancel is not None and cancel.is_set():
                self._publish_status(
                    'trajectory_endpoint_missed',
                    command=command,
                    phase=phase,
                    reason='cancelled',
                    torso_error=last_torso_error,
                    arm_error=last_arm_error,
                )
                return None
            fault = self._payload_hazard_reason(max_age=endpoint_contact_age)
            if fault is not None:
                publish_grasp_fault(fault)
                return None

            measured = self._measured_left_solution()
            last_torso_error = abs(float(measured[0] - target[0]))
            last_arm_error = float(np.max(np.abs(measured[1:] - target[1:])))
            if (
                last_torso_error <= torso_tolerance
                and last_arm_error <= arm_tolerance
            ):
                # Close the callback race between the measured-state read and
                # acceptance of the endpoint.
                fault = self._payload_hazard_reason(
                    max_age=endpoint_contact_age
                )
                if fault is not None:
                    publish_grasp_fault(fault)
                    return None
                return measured

            now_ns = self.get_clock().now().nanoseconds
            if started_ns is None:
                started_ns = now_ns
                wall_deadline = (
                    time.monotonic()
                    + float(getattr(self, 'timeout', 120.0))
                    + 4.0 * settle_timeout
                )
            simulated_timeout = bool(
                now_ns >= started_ns
                and now_ns - started_ns >= required_ns
            )
            wall_timeout = bool(
                wall_deadline is not None
                and time.monotonic() >= wall_deadline
            )
            if simulated_timeout or wall_timeout:
                self._publish_status(
                    'trajectory_endpoint_missed',
                    command=command,
                    phase=phase,
                    reason=(
                        'simulated_settle_timeout'
                        if simulated_timeout
                        else 'wall_settle_timeout'
                    ),
                    settle_timeout=settle_timeout,
                    torso_error=last_torso_error,
                    arm_error=last_arm_error,
                )
                return None
            time.sleep(0.02)

    def _retention_after_leg(self, command: str, phase: str, leg: int) -> bool:
        # A callback from before a short trajectory must not certify retention
        # at its endpoint.  Continuous physical contact publishes frequently,
        # so require a sample close to completion without adding a dwell after
        # every subdivided leg.
        endpoint_contact_age = min(
            0.15,
            float(getattr(self, 'grasp_contact_max_age', 0.15)),
        )
        fault = self._payload_hazard_reason(max_age=endpoint_contact_age)
        retained = fault is None
        if fault is not None:
            self._publish_status(
                'grasp_lost',
                command=command,
                phase=phase,
                leg=leg,
                reason=fault,
            )
        return retained

    def _fresh_retention_probe(
        self,
        command: str,
        phase: str,
        *,
        leg: Optional[int] = None,
        require_new_sample: bool = True,
    ) -> bool:
        """Verify recent contact, optionally resetting its sample epoch first."""
        lock = getattr(self, '_lock', None)
        if require_new_sample:
            if lock is None:
                self._retention_probe_active = True
            else:
                with lock:
                    self._retention_probe_active = True
        try:
            if require_new_sample:
                self._clear_target_contact_samples()
                probe_seconds = min(0.20, 0.5 * self.grasp_contact_max_age)
                settled = self._wait_sim_duration(probe_seconds)
            else:
                # The held-payload monitor has been active throughout
                # navigation.  At its endpoint, a <=150 ms contact snapshot is
                # stronger and faster than adding a stationary reset/dwell
                # while gravity creeps.
                probe_seconds = 0.0
                settled = True
            (
                retained,
                grasp_width,
                left_contact,
                right_contact,
                plausible_width,
            ) = self._pinch_sample(
                max_age=min(0.15, float(self.grasp_contact_max_age)),
            )
            hazard = getattr(self, '_payload_hazard_latched', None)
            if hazard is None and bool(
                getattr(self, '_target_robot_contact_latched', False)
            ):
                hazard = 'payload_robot_contact'
            if hazard is None and not retained:
                hazard = 'contact_lost'
            retained = bool(retained and hazard is None)
            verified = bool(settled and retained and plausible_width)
            fields = {
                'command': command,
                'phase': phase,
                'verified': verified,
                'retained': retained,
                'measured_position': grasp_width,
                'left_contact': left_contact,
                'right_contact': right_contact,
                'bilateral_contact': left_contact and right_contact,
                'plausible_width': plausible_width,
                'settled': settled,
                'contact_probe_seconds': probe_seconds,
            }
            if hazard is not None:
                fields['reason'] = hazard
            if leg is not None:
                fields['leg'] = leg
            self._publish_status('retention_verified', **fields)
            if not verified:
                lost_fields = {'command': command, 'phase': phase}
                if leg is not None:
                    lost_fields['leg'] = leg
                self._publish_status('grasp_lost', **lost_fields)
            return verified
        finally:
            if require_new_sample:
                if lock is None:
                    self._retention_probe_active = False
                else:
                    with lock:
                        self._retention_probe_active = False

    def _execute_retained_arm_legs(
        self,
        legs: Sequence[Tuple[Sequence[float], float, str]],
        command: str,
        *,
        fresh_retention_phases: Sequence[str] = (),
        leg_offset: int = 0,
        initial_pressure_gate=None,
        arm_speed_scale: float = 1.0,
        withdrawal_speed_scale: float = 1.0,
        clearance_timing=None,
        release_clearance=None,
        withdrawal_timing=None,
        transition_stop=None,
    ) -> Tuple[bool, int, bool]:
        """Monitor each carried action; failed motion leaves its leg incomplete."""
        if release_clearance is not None:
            release_clearance.require_execution(self, legs, command,
                arm_speed_scale=arm_speed_scale, leg_offset=leg_offset,
                fresh_retention_phases=fresh_retention_phases, initial_pressure_gate=initial_pressure_gate,
                clearance_timing=clearance_timing, withdrawal_timing=withdrawal_timing,
                withdrawal_speed_scale=withdrawal_speed_scale, transition_stop=transition_stop)
        if transition_stop is not None:
            transition_stop.require_execution(self, legs, command,
                arm_speed_scale=arm_speed_scale, leg_offset=leg_offset,
                fresh_retention_phases=fresh_retention_phases, initial_pressure_gate=initial_pressure_gate,
                clearance_timing=clearance_timing, withdrawal_timing=withdrawal_timing,
                withdrawal_speed_scale=withdrawal_speed_scale)
        if withdrawal_timing is not None:
            validate_withdrawal_half_execution(self, withdrawal_timing, legs, command,
                leg_offset=leg_offset, initial_pressure_gate=initial_pressure_gate,
                arm_speed_scale=arm_speed_scale, withdrawal_speed_scale=withdrawal_speed_scale,
                fresh_retention_phases=fresh_retention_phases, clearance_timing=clearance_timing)
        if clearance_timing is not None:
            validate_bin_clearance_execution(self, clearance_timing, legs, command,
                leg_offset=leg_offset, initial_pressure_gate=initial_pressure_gate,
                withdrawal_speed_scale=withdrawal_speed_scale,
                fresh_retention_phases=fresh_retention_phases)
        if arm_speed_scale != 1.0:
            arm_speed_scale = checked_arm_speed_scale(arm_speed_scale)
        withdrawal_scales = None
        if isinstance(withdrawal_speed_scale, bool):
            raise ValueError('withdrawal speed must be numeric, not Boolean')
        if withdrawal_speed_scale != 1.0:
            try:
                withdrawal_scales = withdrawal_leg_speed_scales(
                    legs, command=command, speed_scale=withdrawal_speed_scale,
                    arm_speed_scale=arm_speed_scale, leg_offset=leg_offset,
                    initial_pressure_gate=initial_pressure_gate)
            except ValueError as exc:
                # PICK is already closed; a bad optional prefix must not retry/open.
                self._publish_status(
                    'withdrawal_timing_admission_rejected', command=command,
                    reason=str(exc), leg_count=len(legs),
                    retained_stop=True, recovery_halted=True,
                )
                raise RuntimeError('pick_recovery_failed') from exc
        for index, (solution, duration, phase) in enumerate(legs):
            release_connector = None
            try:
                effective_speed_scale = (arm_speed_scale if withdrawal_scales is None else
                                         withdrawal_scales[index])
                faster_clearance = (clearance_timing is not None and index == clearance_timing.index)
                timing_duration = (bin_clearance_timing_input(self, clearance_timing,
                    index, solution, duration, phase) if faster_clearance else duration)
                if release_clearance is not None and index == release_clearance.index:
                    release_connector = release_clearance.admit(index, solution, duration, phase)
                    timing_duration = 1.4
                faster_withdrawal = withdrawal_timing is not None and 1 <= index <= 4
                withdrawal_admission = None
                if faster_withdrawal:
                    timing_duration, withdrawal_admission = withdrawal_half_timing_input(
                        self, withdrawal_timing, index, solution, duration, phase)
                command_duration = (timing_duration if effective_speed_scale == 1.0 else
                                    scaled_arm_seconds(timing_duration, effective_speed_scale))
                active_leg = [(solution, command_duration, phase)]
                goal, total_duration = self._make_retained_arm_trajectory_goal(
                    active_leg,
                )
                moved, motion_contact_lost = self._send_retained_arm_trajectory(
                    # The goal/phase clock is scaled; the old watchdog allowance is not.
                    goal, (total_duration if effective_speed_scale == 1.0 and not faster_clearance else float(duration)),
                    active_leg, command,
                    leg_offset=index + leg_offset,
                    **({'velocity_admission': True} if effective_speed_scale != 1.0 or faster_clearance else {}),
                    **({'velocity_headroom': True, 'clearance_admission': clearance_timing}
                       if faster_clearance else {}),
                    **({'velocity_headroom': True, 'release_clearance_admission': release_connector}
                       if release_connector is not None else {}),
                    **({'velocity_headroom': True, 'withdrawal_admission': withdrawal_admission}
                       if faster_withdrawal else {}),
                    **({'initial_pressure_gate': initial_pressure_gate}
                       if index == 0 and initial_pressure_gate is not None else {}),
                    **({'transition_stop': transition_stop}
                       if transition_stop is not None and index == 1 else {}),
                )
            except ArmVelocityAdmissionRejected as exc:
                if withdrawal_scales is not None:
                    # Preserve the veto cause while selecting the mission's abort-only path.
                    raise RuntimeError('pick_recovery_failed') from exc
                raise
            except LiftPressureRejected:
                # This optional pre-send refusal authorizes no recovery/open.
                raise
            except RetainedMotionNotStopped:
                # The controller may still be moving. Do not enter a recovery
                # that opens the hand or sends an arm goal from an assumed stop.
                raise
            except Exception as exc:
                if transition_stop is not None and exc is transition_stop.publication_rejection:
                    raise
                if getattr(self, '_goal_handles', ()):
                    # An exception after acceptance is not a confirmed stop.
                    # Leave the handle registered for cancellation/shutdown and
                    # prevent open/recovery commands while it may still move.
                    self._cancel.set()
                    raise RetainedMotionNotStopped(
                        'retained arm action failed without a confirmed stop'
                    ) from exc
                self._publish_status(
                    'motion_exception',
                    command=command,
                    phase=phase,
                    leg=index + leg_offset,
                    reason=str(exc),
                )
                return False, index, False
            finally:
                if release_connector is not None:
                    release_connector.close()
                if transition_stop is not None and index == 1:
                    transition_stop.close()
            if not moved:
                # A cancelled action can stop anywhere inside this segment.
                # Only a completed action followed by a failed endpoint probe
                # may return index + 1 below.
                return False, index, motion_contact_lost
            if release_connector is not None:
                release_connector.require_endpoint()
            if faster_clearance:
                # A stopped controller result does not prove measured arrival.
                # This added gate aborts without advancing/recovering on a miss.
                require_bin_clearance_endpoint(self, clearance_timing, index)
            if faster_withdrawal:
                require_withdrawal_half_endpoint(self, withdrawal_admission)
            if phase in fresh_retention_phases:
                retained = self._fresh_retention_probe(
                    command,
                    phase,
                    leg=index + leg_offset,
                )
            else:
                retained = self._retention_after_leg(
                    command,
                    phase,
                    index + leg_offset,
                )
            if not retained:
                return False, index + 1, True
            if transition_stop is not None and index == 0:
                # Success and original retention precede the added no-motion wait.
                transition_stop.qualify()
            if release_clearance is not None:
                release_clearance.retained(index)
        return True, len(legs), False

    @staticmethod
    def _best_effort(operation: Callable[[], bool]) -> bool:
        """Return False for a recovery operation that raises or fails."""
        try:
            return bool(operation())
        except Exception:
            return False

    def _recover_closed_pick(
        self,
        *,
        cause: str,
        remaining_carried_legs: Sequence[Tuple[Sequence[float], float, str]],
        unloaded_recovery_route: Sequence[Sequence[float]],
        lower_shelf_pick: bool = False,
    ) -> bool:
        """Follow the checked route, release on loss or at its terminal, recover."""
        lock = getattr(self, '_lock', None)

        def halt_if_robot_contact() -> None:
            if lock is None:
                robot_contact = bool(
                    getattr(self, '_target_robot_contact_latched', False)
                )
            else:
                with lock:
                    robot_contact = bool(
                        getattr(self, '_target_robot_contact_latched', False)
                    )
            if cause == 'payload_robot_contact' or robot_contact:
                # A payload already touching the robot has no checked recovery
                # transform.  Moving another joint or opening the gripper can
                # turn a contained contact into a drop across the arm, so stop
                # exactly where the collision was detected.
                self._publish_status(
                    'carried_recovery',
                    command='pick',
                    cause='payload_robot_contact',
                    retained_before_recovery=False,
                    gripper_opened=False,
                    recovery_succeeded=False,
                    recovery_halted=True,
                    retained_stop=True,
                )
                raise RuntimeError('pick_payload_lost')

        halt_if_robot_contact()
        if cause in (
            'final_grasp_settle_failed',
            'final_grasp_width_invalid',
        ):
            # A failed clock/cancellation dwell or an implausible jaw sample
            # does not prove that the payload is absent.  No recovery transform
            # is valid for that ambiguous state, so keep the gripper closed and
            # stop instead of converting uncertainty into a drop over the arm.
            retained = self._best_effort(self._target_contact_recent)
            self._publish_status(
                'carried_recovery',
                command='pick',
                cause=cause,
                retained_before_recovery=retained,
                gripper_opened=False,
                recovery_succeeded=False,
                recovery_halted=True,
                retained_stop=True,
            )
            raise RuntimeError('pick_recovery_failed')
        if cause in ('motion_failed', 'torso_motion_failed'):
            retained = self._best_effort(
                lambda: self._fresh_retention_probe(
                    'pick',
                    'motion_failure_recovery',
                )
            )
            halt_if_robot_contact()
            if retained:
                # An aborted controller action leaves the actual joint state
                # somewhere inside the commanded segment.  Even with a fresh
                # pinch, replaying a route from its planned endpoint is not a
                # validated recovery.  Keep holding the payload and stop.
                self._publish_status(
                    'carried_recovery',
                    command='pick',
                    cause=cause,
                    retained_before_recovery=True,
                    gripper_opened=False,
                    recovery_succeeded=False,
                    recovery_halted=True,
                    retained_stop=True,
                )
                raise RuntimeError('pick_recovery_failed')
            if getattr(self, '_gravity_supported_payload', False):
                # Missing contact above the arm is not proof that the supported
                # payload has fallen clear.  Opening here can create the very
                # robot collision the recovery is meant to prevent.
                self._publish_status(
                    'carried_recovery',
                    command='pick',
                    cause=cause,
                    retained_before_recovery=False,
                    gripper_opened=False,
                    recovery_succeeded=False,
                    recovery_halted=True,
                    retained_stop=True,
                )
                raise RuntimeError('pick_payload_lost')
            halt_if_robot_contact()
            gripper_opened = self._best_effort(self._open_gripper)
            self._publish_status(
                'carried_recovery',
                command='pick',
                cause=cause,
                retained_before_recovery=False,
                gripper_opened=gripper_opened,
                recovery_succeeded=False,
                recovery_halted=True,
            )
            raise RuntimeError('pick_payload_lost')
        confirmed_contact_loss = cause in (
            'contact_lost',
            'contact_lost_after_torso',
            'cradle_contact_verification_failed',
            'final_grasp_contact_lost',
        )
        if (
            confirmed_contact_loss
            and getattr(self, '_gravity_supported_payload', False)
        ):
            # A missing supported-contact callback is ambiguous while the book
            # is above the arm: opening can turn a safely cradled payload into
            # a collision.  Hold the commanded jaw position and stop instead.
            self._publish_status(
                'carried_recovery',
                command='pick',
                cause=cause,
                retained_before_recovery=False,
                gripper_opened=False,
                recovery_succeeded=False,
                recovery_halted=True,
                retained_stop=True,
            )
            raise RuntimeError('pick_payload_lost')
        retained = (
            False
            if confirmed_contact_loss
            else self._best_effort(self._target_contact_recent)
        )
        retained_before_recovery = retained
        gripper_opened = False
        halt_if_robot_contact()
        if not retained:
            gripper_opened = self._best_effort(self._open_gripper)
            # Once contact has been lost, the rigid book-to-gripper transform
            # used to validate every remaining arm segment is no longer true.
            # Continuing that route caused the loose live target to strike a
            # second proximal arm link.  Stop the arm and surface a terminal
            # payload-loss result; a displaced book is not an empty-grasp
            # failure that the mission can safely retry.
            self._publish_status(
                'carried_recovery',
                command='pick',
                cause=cause,
                retained_before_recovery=False,
                gripper_opened=gripper_opened,
                recovery_succeeded=False,
                recovery_halted=True,
            )
            raise RuntimeError('pick_payload_lost')
        # The lower-shelf plan has no admitted loaded recovery from an
        # arbitrary failure state.  Its numerical HOME-height arm route does
        # not check a loaded torso sweep or a supported release.  Preserve the
        # measured retained state until a complete recovery can be admitted.
        if lower_shelf_pick:
            self._publish_status(
                'carried_recovery',
                command='pick',
                cause=cause,
                reason='lower_shelf_recovery_certificate_missing',
                retained_before_recovery=True,
                gripper_opened=False,
                recovery_succeeded=False,
                recovery_halted=True,
                retained_stop=True,
            )
            raise RuntimeError('pick_recovery_failed')
        recovery_ok = retained or gripper_opened
        for leg, (solution, duration, phase) in enumerate(
            remaining_carried_legs
        ):
            if not recovery_ok:
                break
            halt_if_robot_contact()
            if not self._best_effort(
                lambda q=solution, seconds=duration: self._move_arm_solution(
                    q,
                    seconds,
                )
            ):
                recovery_ok = False
                break
            if retained and not self._best_effort(
                lambda p=phase, index=leg: self._retention_after_leg(
                    'pick',
                    f'recovery_{p}',
                    index,
                )
            ):
                retained = False
                halt_if_robot_contact()
                supported = bool(
                    getattr(self, '_gravity_supported_payload', False)
                )
                gripper_opened = (
                    False
                    if supported
                    else self._best_effort(self._open_gripper)
                )
                self._publish_status(
                    'carried_recovery',
                    command='pick',
                    cause='contact_lost_during_recovery',
                    retained_before_recovery=retained_before_recovery,
                    gripper_opened=gripper_opened,
                    recovery_succeeded=False,
                    recovery_halted=True,
                    retained_stop=supported,
                )
                raise RuntimeError('pick_payload_lost')
        if recovery_ok:
            halt_if_robot_contact()
            recovery_ok = self._best_effort(
                lambda: self._move_torso(float(HOME[0]), 2.0)
            )
        if recovery_ok and retained:
            halt_if_robot_contact()
            gripper_opened = self._best_effort(self._open_gripper)
        if recovery_ok and gripper_opened:
            for solution in unloaded_recovery_route:
                if not self._best_effort(
                    lambda q=solution: self._move_arm_solution(q, 2.2)
                ):
                    recovery_ok = False
                    break
        if not gripper_opened:
            recovery_ok = False
        self._publish_status(
            'carried_recovery',
            command='pick',
            cause=cause,
            retained_before_recovery=retained_before_recovery,
            gripper_opened=gripper_opened,
            recovery_succeeded=recovery_ok,
        )
        if not recovery_ok:
            raise RuntimeError('pick_recovery_failed')
        return False

    def _return_arm_home(
        self,
        transition_waypoints: Sequence[Sequence[float]] = (),
        *,
        waypoint_duration: float = 2.2,
    ) -> bool:
        if getattr(self, '_empty_arm_staged', False):
            return False
        for waypoint in reversed(transition_waypoints):
            if not self._move_arm_solution(waypoint, waypoint_duration):
                return False
        if not self._follow(self.arm_client, ARM_JOINTS, HOME[1:], 2.8):
            return False
        return self._move_torso(float(HOME[0]), 2.0)

    def _recover_unloaded_pick(
        self,
        cartesian_solutions: Sequence[Sequence[float]],
        transition_waypoints: Sequence[Sequence[float]],
    ) -> bool:
        """Retrace Cartesian extraction, then the separate transition route."""
        for solution in reversed(cartesian_solutions[:-1]):
            if not self._move_arm_solution(solution, 0.65):
                return False
        return self._return_arm_home(transition_waypoints)

    def _recover_ambiguous_grasp(
        self,
        attached_corners: Sequence[Sequence[float]],
        cartesian_solutions: Sequence[Sequence[float]],
        transition_waypoints: Sequence[Sequence[float]],
    ) -> bool:
        """Require a successful reopen before treating a failed close as empty."""
        self._held_book_corners = np.asarray(attached_corners, dtype=float).copy()
        lock = getattr(self, '_lock', None)
        if lock is None:
            robot_contact = bool(
                getattr(self, '_target_robot_contact_latched', False)
            )
        else:
            with lock:
                robot_contact = bool(
                    getattr(self, '_target_robot_contact_latched', False)
                )
        if robot_contact:
            # A failed close can itself be caused by the fresh retention probe
            # observing book-to-robot contact.  That is a known collision, not
            # an empty grasp: opening here can drop the book across the arm.
            return self._recover_closed_pick(
                cause='payload_robot_contact',
                remaining_carried_legs=(),
                unloaded_recovery_route=(),
            )
        if not self._best_effort(self._open_gripper):
            raise RuntimeError('pick_recovery_failed')
        recovered = self._best_effort(
            lambda: self._recover_unloaded_pick(
                cartesian_solutions,
                transition_waypoints,
            )
        )
        if not recovered:
            raise RuntimeError('pick_recovery_failed')
        return False

    def _return_from_bin(
        self,
        cartesian_solutions: Sequence[Sequence[float]],
        carried_transition_waypoints: Sequence[Sequence[float]],
        carried_start: Sequence[float],
        unloaded_home_waypoints: Sequence[Sequence[float]],
        *,
        direct_empty_home: Optional[Sequence[Sequence[float]]] = None,
        arm_speed_scale: float = 1.0,
        keep_torso_height: bool = False,
    ) -> bool:
        """Retrace the bin approach, then execute a prechecked unloaded fold."""
        from .release_only_place_planning import reject_return
        reject_return(unloaded_home_waypoints)
        reject_return(direct_empty_home)
        keep_torso_height = checked_raised_place_finish_enabled(keep_torso_height)
        if keep_torso_height and (direct_empty_home is None
                or not getattr(self, 'table_scene_required', False)):
            raise RuntimeError('raised PLACE finish requires the measured direct return')
        if arm_speed_scale != 1.0:
            arm_speed_scale = checked_arm_speed_scale(arm_speed_scale)
        endpoint_wait = None
        previous = None
        if direct_empty_home is not None and getattr(self, 'table_scene_required', False):
            previous = np.asarray(cartesian_solutions[-1], dtype=float)
            from .empty_pickup_collision import (
                joint_velocity_limits, measured_context, wait_for_geometry_endpoint,
            )
            scene_reference = getattr(self, '_active_place_scene_reference', None)
            if scene_reference is None:
                raise RuntimeError('direct empty return requires its checked scene context')
            def check_return_context():
                _require_place_contact_clear(self)
                measured_scene_context(self, scene_reference)
            check_return_context()
            joints, _ = measured_context(self)
            right = np.asarray([joints[n] for n in RIGHT_ARM_JOINTS])
            head = np.asarray([joints[n] for n in HEAD_JOINTS])
            urdf = Path(get_package_share_directory('erc_description')) / 'urdf/tiago_pro.urdf'
            limits = joint_velocity_limits(urdf)
            def endpoint_wait(first, goal, phase):
                return wait_for_geometry_endpoint(self, first, goal,
                    right=right, head=head, velocity_limits=limits,
                    aperture=self.gripper_open, phase=phase, command='place',
                    context_check=check_return_context)
        for solution in reversed(cartesian_solutions[:-1]):
            if not self._move_arm_solution(solution, 0.65,
                    **({'trajectory_duration': scaled_arm_seconds(.65, arm_speed_scale)}
                       if arm_speed_scale != 1.0 else {})):
                return False
            if endpoint_wait is not None:
                endpoint_wait(previous, solution, 'empty_return_cartesian')
            previous = np.asarray(solution, dtype=float)
        if direct_empty_home is not None:
            duration = max(2.8, 4.0*float(np.max(np.abs(
                np.asarray(cartesian_solutions[0])[1:]-HOME[1:]))))
            return self._execute_unloaded_home(direct_empty_home, final_arm_duration=duration,
                **({'keep_torso_height': True} if keep_torso_height else {}),
                **({'arm_speed_scale': arm_speed_scale} if arm_speed_scale != 1.0 else {}),
                **({'endpoint_wait': endpoint_wait, 'endpoint_start': previous}
                   if endpoint_wait is not None else {}))
        for waypoint in reversed(carried_transition_waypoints):
            if not self._move_arm_solution(waypoint, 0.8):
                return False
        if not self._move_arm_solution(carried_start, 0.8):
            return False
        return self._execute_unloaded_home(unloaded_home_waypoints)

    def _execute_unloaded_home(
        self,
        unloaded_home_waypoints: Sequence[Sequence[float]],
        *,
        final_arm_duration: float = 2.8,
        endpoint_wait: Optional[Callable] = None,
        endpoint_start: Optional[Sequence[float]] = None,
        arm_speed_scale: float = 1.0,
        keep_torso_height: bool = False,
    ) -> bool:
        """Execute a prechecked unloaded route, normally lowering the torso."""
        from .release_only_place_planning import reject_return
        reject_return(unloaded_home_waypoints)
        keep_torso_height = checked_raised_place_finish_enabled(keep_torso_height)
        if keep_torso_height and not callable(endpoint_wait):
            raise RuntimeError('raised PLACE finish requires measured return endpoints')
        if arm_speed_scale != 1.0:
            arm_speed_scale = checked_arm_speed_scale(arm_speed_scale)
        previous = None
        if endpoint_wait is not None:
            previous = np.asarray(endpoint_start, dtype=float)
            if previous.shape != (8,) or not np.all(np.isfinite(previous)):
                raise ValueError('measured empty return requires its checked start')
        for waypoint in unloaded_home_waypoints:
            if not self._move_arm_solution(waypoint, 2.2,
                    **({'trajectory_duration': scaled_arm_seconds(2.2, arm_speed_scale)}
                       if arm_speed_scale != 1.0 else {})):
                return False
            if endpoint_wait is not None:
                endpoint_wait(previous, waypoint, 'empty_return_waypoint')
                previous = np.asarray(waypoint, dtype=float)
        if not self._follow(self.arm_client, ARM_JOINTS, HOME[1:], final_arm_duration,
                **({'trajectory_duration': scaled_arm_seconds(final_arm_duration, arm_speed_scale)}
                   if arm_speed_scale != 1.0 else {})):
            return False
        if endpoint_wait is not None:
            home_at_height = HOME.copy()
            home_at_height[0] = previous[0]
            endpoint_wait(previous, home_at_height, 'empty_return_arm_home')
            previous = home_at_height
        if keep_torso_height:
            return True
        if not self._move_torso(float(HOME[0]), 2.0):
            return False
        if endpoint_wait is not None:
            endpoint_wait(previous, HOME, 'empty_return_torso_home')
        return True

    def _recover_closed_place(
        self,
        *,
        cause: str,
        remaining_approach_legs: Sequence[Tuple[Sequence[float], float, str]],
        cartesian_solutions: Sequence[Sequence[float]],
        carried_transition_waypoints: Sequence[Sequence[float]],
        carried_start: Sequence[float],
        unloaded_home_waypoints: Sequence[Sequence[float]],
        at_carried_start: bool = False,
        ensure_place_torso: bool = False,
        release_allowed: bool = True,
    ) -> bool:
        """Recover only when release is safe; otherwise stop while closed."""
        from .release_only_place_planning import reject_return
        reject_return(unloaded_home_waypoints)
        if cause in ('contact_lost', 'contact_lost_after_torso'):
            self._publish_status(
                'carried_recovery',
                command='place',
                cause=cause,
                gripper_opened=False,
                recovery_succeeded=False,
                recovery_halted=True,
                retained_stop=True,
            )
            # Contact loss is ambiguous until the commanded release pose is
            # reached.  Do not open or move through potentially loose cargo.
            raise RuntimeError('place_payload_lost')
        if not release_allowed:
            self._publish_status(
                'carried_recovery',
                command='place',
                cause=cause,
                gripper_opened=False,
                recovery_succeeded=False,
                retained_stop=True,
            )
            raise RuntimeError('place_recovery_failed')
        gripper_opened = self._best_effort(self._open_gripper)
        recovery_ok = gripper_opened
        if recovery_ok and ensure_place_torso:
            recovery_ok = self._best_effort(
                lambda: self._move_torso(float(carried_start[0]), 2.2)
            )
        if recovery_ok and at_carried_start:
            recovery_ok = self._best_effort(
                lambda: self._execute_unloaded_home(unloaded_home_waypoints)
            )
        elif recovery_ok:
            for solution, duration, _ in remaining_approach_legs:
                if not self._best_effort(
                    lambda q=solution, seconds=duration: self._move_arm_solution(
                        q,
                        seconds,
                    )
                ):
                    recovery_ok = False
                    break
            if recovery_ok:
                recovery_ok = self._best_effort(
                    lambda: self._return_from_bin(
                        cartesian_solutions,
                        carried_transition_waypoints,
                        carried_start,
                        unloaded_home_waypoints,
                    )
                )
        self._publish_status(
            'carried_recovery',
            command='place',
            cause=cause,
            gripper_opened=gripper_opened,
            recovery_succeeded=recovery_ok,
        )
        if not recovery_ok:
            raise RuntimeError('place_recovery_failed')
        return False

    def _solve_target(self, position: Sequence[float]) -> Tuple[np.ndarray, float]:
        seeds: List[np.ndarray] = [self._current_seed(), OFFER, HOME, PREGRASP]
        best_score = float('inf')
        for rotation in shelf_grasp_orientations(float(position[2])):
            solution, score = self.chain.solve(
                pose_matrix(position, rotation),
                seeds,
                position_tolerance=self.position_tolerance,
                orientation_tolerance=self.orientation_tolerance,
                max_iterations=180,
            )
            best_score = min(best_score, score)
            if solution is not None:
                return solution, score
        if not np.isfinite(best_score):
            best_score = float('inf')
        if best_score == float('inf'):
            raise RuntimeError(f'IK did not converge for target {np.round(position, 3)}')
        raise RuntimeError(
            'IK did not converge for target '
            f'{np.round(position, 3)} (best score {best_score:.4f})'
        )

    def _grasp_depth_for_height(self, height: float) -> float:
        if float(height) >= 1.42:
            return self.top_row_grasp_depth_offset
        return self.grasp_depth_offset

    def _loaded_clearance_index(
        self,
        top_row: bool,
        solutions: Sequence[Sequence[float]],
    ) -> int:
        """Select the final loaded extraction pose for a solved pick path."""
        return 1 if top_row and len(solutions) > 1 else 0

    def _pick_path_accuracy(
        self,
        positions: Sequence[Sequence[float]],
        rotation: np.ndarray,
        solutions: Sequence[Sequence[float]],
    ) -> Dict:
        """Report actual solved FK, including alignment along the full pick path."""
        if not solutions or len(positions) != len(solutions):
            raise RuntimeError('Pick accuracy check requires the complete Cartesian path')
        position_errors, orientation_errors = [], []
        actual = None
        for position, solution in zip(positions, solutions):
            actual = np.asarray(self.chain.forward(solution), dtype=float)
            desired = pose_matrix(position, rotation)
            if actual.shape != (4, 4) or not np.all(np.isfinite(actual)):
                raise RuntimeError('Pick accuracy check received invalid FK')
            residual = self.chain.pose_error(actual, desired)
            position_errors.append(float(np.linalg.norm(residual[:3])))
            orientation_errors.append(float(np.linalg.norm(residual[3:])))
        if not all(np.isfinite(value) for value in (*position_errors, *orientation_errors)):
            raise RuntimeError('Pick accuracy check received nonfinite residuals')
        return {
            'actual_grasp_position': actual[:3, 3].tolist(),
            'actual_grasp_orientation': actual[:3, :3].tolist(),
            'grasp_position_error_base_m': (
                actual[:3, 3] - np.asarray(positions[-1], dtype=float)
            ).tolist(),
            'grasp_position_error_m': position_errors[-1],
            'grasp_orientation_error_rad': orientation_errors[-1],
            'approach_max_position_error_m': max(position_errors),
            'approach_max_orientation_error_rad': max(orientation_errors),
        }

    def _lift_first_measurements(self, reference=None):
        """Fresh measured geometry and a stationary, unchanged registered base."""
        geometry = self._shelf_cradle_geometry
        # The measured master drives a screw on another URDF branch. It is
        # not an ancestor of the nine collision meshes: their hinge chains
        # depend on it through <mimic>, which local_surfaces resolves explicitly.
        finger_names = sorted({'gripper_left_finger_joint', *(
            name for chain in geometry.chains.values() for name in chain.active_names
            if name.startswith('gripper_left_'))})
        with self._lock:
            joints = dict(self.joints)
            stamps = dict(getattr(self, '_joint_stamps_ns', {}))
            sample = getattr(self, '_staging_odom', None)
            odom = dict(sample) if sample is not None else None
        now = int(self.get_clock().now().nanoseconds)
        passive_names = [name for name in finger_names if name != 'gripper_left_finger_joint']
        # Official /joint_states reports actuated joints, not all URDF mimics.
        # Any available hinge sample must be valid; absent hinges remain an
        # explicitly modeled linkage, never fabricated sensor measurements.
        reported_passive = [name for name in passive_names if name in joints]
        names = (*IK_JOINTS, *RIGHT_ARM_JOINTS, *HEAD_JOINTS,
                 'gripper_left_finger_joint', *reported_passive)
        for name in names:
            if (name not in joints or not math.isfinite(joints[name])
                    or not -.05e9 <= now-stamps.get(name, 0) <= .35e9):
                raise RuntimeError(f'lift-first joint measurement stale: {name}')
        if odom is None or not -.05e9 <= now-odom.get('stamp_ns', 0) <= .35e9:
            raise RuntimeError('lift-first odometry is stale')
        pose = np.asarray(odom.get('pose'), dtype=float)
        speeds = np.asarray([odom.get('linear_speed'), odom.get('angular_speed')], dtype=float)
        if (pose.shape != (3,) or not np.all(np.isfinite(pose))
                or not np.all(np.isfinite(speeds)) or np.any(speeds < 0)
                or speeds[0] > .005 or speeds[1] > .008):
            raise RuntimeError('lift-first base is not measured stationary')
        if reference is not None:
            previous = reference['base_pose']
            yaw_delta = math.atan2(math.sin(pose[2]-previous[2]),
                                   math.cos(pose[2]-previous[2]))
            if np.linalg.norm(pose[:2]-previous[:2]) > .002 or abs(yaw_delta) > .005:
                raise RuntimeError('lift-first registered base moved')
            for name in (*RIGHT_ARM_JOINTS, *HEAD_JOINTS):
                if abs(joints[name]-reference['joints'][name]) > .001:
                    raise RuntimeError(f'lift-first collision geometry moved: {name}')
        complete_passive = len(reported_passive) == len(passive_names)
        measured_names = finger_names if complete_passive else ['gripper_left_finger_joint']
        fingers = {name: float(joints[name]) for name in measured_names}
        modeled_names = [] if complete_passive else passive_names
        return dict(joints=joints, fingers=fingers,
                    left=np.asarray([joints[name] for name in IK_JOINTS]),
                    base_pose=pose.copy(), stamp_ns=now,
                    modeled_finger_joints=modeled_names,
                    reported_passive_joints=reported_passive,
                    finger_geometry_basis=('measured_master_and_all_passive_joints'
                        if complete_passive else 'measured_master_with_official_urdf_mimics'),
                    modeled_tool_allowance_m=.0 if complete_passive else .005)

    @staticmethod
    def _check_lift_collision_context(current, context):
        """Compare fresh parked geometry with the snapshot actually checked."""
        if not isinstance(context, dict):
            raise RuntimeError('lift-first measured collision context is absent')
        for key, names in (('right_positions', RIGHT_ARM_JOINTS),
                           ('head_positions', HEAD_JOINTS)):
            expected = np.asarray(context.get(key), dtype=float)
            actual = np.asarray([current['joints'][name] for name in names], dtype=float)
            if (expected.shape != (len(names),) or not np.all(np.isfinite(expected))
                    or not np.all(np.isfinite(actual))):
                raise RuntimeError('lift-first measured collision context is invalid')
            for name, delta in zip(names, np.abs(actual-expected)):
                if delta > .001:
                    raise RuntimeError(f'lift-first collision snapshot moved: {name}')

    def _recheck_lift_first_geometry(self, front, grasp, lift_plan, bay, reference):
        """Recheck using measured master and optional measured passive linkage."""
        from .lift_first_extraction import validate_lift_first_route
        measured = self._lift_first_measurements(reference)
        geometry_context = dict(
            right_positions=tuple(float(measured['joints'][name]) for name in RIGHT_ARM_JOINTS),
            head_positions=tuple(float(measured['joints'][name]) for name in HEAD_JOINTS),
            stamp_ns=measured['stamp_ns'],
        )
        ManipulationNode._check_lift_collision_context(measured, geometry_context)
        error = self.chain.pose_error(self.chain.forward(measured['left']),
                                      self.chain.forward(grasp))
        if (np.linalg.norm(error[:3]) > self.pick_position_tolerance
                or np.linalg.norm(error[3:]) > self.pick_orientation_tolerance
                or abs(measured['left'][0]-grasp[0]) > .0005
                or np.max(np.abs(measured['left'][1:]-np.asarray(grasp)[1:])) > .01):
            raise RuntimeError('lift-first measured grasp is outside planned precision')
        self._publish_status('lift_first_geometry_started', command='pick',
            geometry_context=geometry_context, legs=len(lift_plan.route),
            aperture=measured['fingers']['gripper_left_finger_joint'])
        checked = run_pickup_geometry(
            self, validate_lift_first_route, front, measured['left'], lift_plan.route,
            scene_reference=reference, bay=bay,
            aperture=measured['fingers']['gripper_left_finger_joint'],
            finger_positions=measured['fingers'],
            lift_m=self.lift_first_extraction_lift_m,
            attached_corners=lift_plan.attached_corners,
            modeled_tool_allowance_m=measured['modeled_tool_allowance_m'],
            right_positions=geometry_context['right_positions'],
            head_positions=geometry_context['head_positions'],
        )
        # Mesh checking may take time. Reject geometry drift instead of reusing
        # the old snapshot as though it were a new sensor observation.
        current = self._lift_first_measurements(reference)
        ManipulationNode._check_lift_collision_context(current, geometry_context)
        if current['finger_geometry_basis'] != measured['finger_geometry_basis']:
            raise RuntimeError('lift-first finger feedback availability changed during preflight')
        if np.max(np.abs(current['left']-measured['left'])) > .0002:
            raise RuntimeError('lift-first arm moved during loaded preflight')
        master_tolerance = (self.adaptive_endpoint_tolerance if _stock_diagnostic(self) else .00001)
        master_geometry = dict(
            geometry_reference_master_position_m=measured['fingers']['gripper_left_finger_joint'],
            master_comparison_reference_position_m=measured['fingers']['gripper_left_finger_joint'],
            current_master_position_m=current['fingers']['gripper_left_finger_joint'],
            master_geometry_tolerance_m=master_tolerance,
            master_geometry_scope='nominal sampled geometry plus engineering admission tolerance; no geometry recomputation or passive deflection proof')
        for name, value in measured['fingers'].items():
            tolerance = master_tolerance if name == 'gripper_left_finger_joint' else .001
            if abs(current['fingers'][name]-value) > tolerance:
                error = RuntimeError(f'lift-first finger moved during loaded preflight: {name}')
                error.master_geometry = master_geometry
                raise error
        self._publish_status('lift_first_measured_geometry_verified', command='pick',
            aperture=current['fingers']['gripper_left_finger_joint'],
            measured_joint_stamp_ns=current['stamp_ns'],
            finger_geometry_basis=current['finger_geometry_basis'],
            modeled_finger_joints=current['modeled_finger_joints'],
            measured_finger_joints=sorted(current['fingers']),
            geometry_context=geometry_context,
            **master_geometry, **checked)
        current['geometry_context'] = geometry_context
        current['geometry_reference_master_position_m'] = measured['fingers']['gripper_left_finger_joint']
        return current

    def _pick(self, payload: Optional[Dict] = None) -> bool:
        interval_enabled = bool(getattr(self, 'preclose_aperture_geometry_enabled', False))
        if interval_enabled:
            from .preclose_aperture_geometry import (
                invalidate, prepare as prepare_interval, reuse as reuse_interval, check_after_probe)
            invalidate(self)
            if not (self.lift_first_extraction_enabled and self.fine_gripper_close_enabled):
                raise RuntimeError('preclose aperture option requires lift-first and fine closure')
        if getattr(self, '_held_book_corners', None) is not None:
            raise RuntimeError('pick requested while a book may still be held')
        self._cached_post_retreat_plan = None
        lift_enabled = bool(getattr(self, 'lift_first_extraction_enabled', False))
        lift_plan, lift_bay, lift_reference = None, None, None
        if lift_enabled:
            from .shelf_bay_context import decode_bay_context
            from .shelf_cradle_geometry import ShelfCradleGeometry
            with self._lock:
                registered_odom = getattr(self, '_staging_odom', None)
                registered_base = np.asarray(
                    registered_odom.get('pose') if isinstance(registered_odom, dict) else None,
                    dtype=float,
                ).copy()
            if registered_base.shape != (3,) or not np.all(np.isfinite(registered_base)):
                raise RuntimeError('lift-first shelf registration needs measured base pose')
            lift_bay = decode_bay_context(self, payload)
            if getattr(self, '_shelf_cradle_geometry', None) is None:
                urdf = Path(get_package_share_directory('erc_description')) / 'urdf' / 'tiago_pro.urdf'
                self._shelf_cradle_geometry = ShelfCradleGeometry(urdf, get_package_share_directory, immutable_local=True)
            lift_reference = self._lift_first_measurements()
            registered_yaw_delta = math.atan2(
                math.sin(lift_reference['base_pose'][2]-registered_base[2]),
                math.cos(lift_reference['base_pose'][2]-registered_base[2]))
            if (np.linalg.norm(lift_reference['base_pose'][:2]-registered_base[:2]) > .002
                    or abs(registered_yaw_delta) > .005):
                raise RuntimeError('lift-first base moved during shelf decoding')
        front = self._wait_for_perception_point('latest_book')
        if getattr(self, '_empty_arm_staged', False):
            from .empty_arm_preparation import (
                angle_error, check_empty_stationary, point_in_pose,
            )
            certificate = getattr(self, '_empty_arm_preparation', None)
            if not certificate:
                raise RuntimeError('staged pick has no preparation certificate')
            measured, odom = check_empty_stationary(self)
            goal = certificate['final_goal']
            if (np.linalg.norm(np.asarray(odom['pose'][:2]) - goal[:2]) > .018
                    or abs(angle_error(odom['pose'][2], goal[2])) > .020):
                raise RuntimeError('staged pick base does not match the prepared goal')
            if any(abs(measured[name] - expected) > .008
                   for name, expected in certificate['prepared_joints'].items()
                   if name in IK_JOINTS):
                raise RuntimeError('staged pick arm changed after preparation')
            expected_front = point_in_pose(certificate['book_odom'], odom['pose'])
            if np.linalg.norm(front - expected_front) > .035:
                raise RuntimeError('reacquired book changed after empty-arm preparation')
        if not (0.30 < front[0] < 1.30 and abs(front[1]) < 0.70 and 0.45 < front[2] < 1.85):
            raise RuntimeError(f'book point outside safe workspace: {np.round(front, 3)}')
        top_row = float(front[2]) >= 1.42
        pick_torso_height = self.pick_torso_height
        lower_pick_plan = None
        depth_offset = self._grasp_depth_for_height(float(front[2]))
        grasp = front.copy()
        grasp[0] += depth_offset
        grasp_vertical_offset = (
            self.top_row_grasp_vertical_offset if top_row else 0.0
        )
        grasp_lateral_offset = (
            self.top_row_grasp_lateral_offset if top_row else 0.0
        )
        grasp[1] += grasp_lateral_offset
        grasp[2] += grasp_vertical_offset
        pregrasp = grasp.copy()
        pregrasp[0] -= self.pregrasp_offset
        clearance = pregrasp.copy()
        clearance[0] = (
            self.top_row_cartesian_clearance
            if top_row
            else self.cartesian_clearance
        )
        loaded_clearance_lift = (
            self.top_row_loaded_clearance_lift if top_row else 0.0
        )
        # Keep the top-row book on the bay floor until a mechanically verified
        # support handoff has been completed.  Retry 26 showed that lifting the
        # still-vertical pinch before that handoff lets the book fall during
        # the first few centimetres of base retreat.  The parameter remains a
        # bounded diagnostic override, but production config deliberately uses
        # zero lift.
        clearance[2] += loaded_clearance_lift
        if pregrasp[0] <= clearance[0]:
            raise RuntimeError(
                f'book is too close for staged approach: {np.round(front, 3)}'
            )
        positions = [clearance]
        positions.extend(
            self._interpolate_positions(clearance, pregrasp, self.cartesian_step)
        )
        positions.extend(
            self._interpolate_positions(pregrasp, grasp, self.cartesian_step)
        )
        rotations = shelf_pinch_orientations(float(grasp[2]))
        # The narrow book needs substantially tighter alignment than transport
        # waypoints. Keep the same endpoint-first/OFFER branch, and preserve
        # these tolerances throughout the path that is reversed while loaded.
        pick_position_tolerance = getattr(self, 'pick_position_tolerance', 0.0005)
        pick_orientation_tolerance = getattr(self, 'pick_orientation_tolerance', 0.01)
        # Ordinary empty pickup previously checked only link-origin bounds
        # during setup. Keep the opt-in closed-finger preparation path intact;
        # the normal open-hand route now checks body, screen and all tool meshes.
        empty_setup_guard = None
        empty_setup_options = {}
        if not (top_row and getattr(self, 'shelf_side_cradle_enabled', False)):
            from .empty_pickup_collision import EmptyPickupCollision
            empty_setup_guard = EmptyPickupCollision.capture(self)
            empty_elevated = empty_setup_guard.start.copy()
            empty_elevated[0] = pick_torso_height
            empty_setup_options = dict(
                transition_start=empty_elevated,
                transition_edge_validator=empty_setup_guard.retracted_edge,
                setup_transition_planner=empty_setup_guard.plan_transition,
                candidate_validator=empty_setup_guard.candidate,
            )
        if (not top_row and lift_enabled
                and getattr(self, 'lower_shelf_pick_enabled', False)):
            from .lower_shelf_pick import plan_lower_shelf_pick
            from .lift_first_extraction import plan_lift_first_extraction

            def plan_lower_lift(grasp_solution, extraction_solutions, *, post_retreat_plan=None):
                self._lift_first_measurements(lift_reference)
                return run_pickup_geometry(
                    self, plan_lift_first_extraction, front, grasp_solution,
                    extraction_solutions, scene_reference=lift_reference,
                    **({'post_retreat_plan': post_retreat_plan} if post_retreat_plan is not None else {}),
                    bay=lift_bay, aperture=float(self.carried_book_dimensions[1]),
                    lift_m=self.lift_first_extraction_lift_m,
                    modeled_tool_allowance_m=.005,
                )

            lower_pick_plan = plan_lower_shelf_pick(
                self, front, empty_guard=empty_setup_guard,
                lift_planner=plan_lower_lift, bay=lift_bay,
            )
            positions = lower_pick_plan.positions
            grasp = lower_pick_plan.grasp
            rotations = lower_pick_plan.rotations
            solutions = lower_pick_plan.solutions
            orientation_index = lower_pick_plan.orientation_index
            path_score = lower_pick_plan.path_score
            transition_waypoints = lower_pick_plan.transition_waypoints
            pick_torso_height = lower_pick_plan.pick_torso_height
            depth_offset = lower_pick_plan.grasp_depth_offset
            grasp_lateral_offset = lower_pick_plan.grasp_lateral_offset
            grasp_vertical_offset = lower_pick_plan.grasp_vertical_offset
            loaded_clearance_lift = 0.0
            empty_elevated = empty_setup_guard.start.copy()
            empty_elevated[0] = pick_torso_height
        else:
            # This empty-only epoch is completely closed before any loaded pool
            # or motion. Disabled/custom paths keep their ordinary serial checker.
            with empty_pickup_geometry_scope(self, empty_setup_guard):
                if empty_setup_guard is not None:
                    if not empty_setup_guard.opening() or not empty_setup_guard.edge(
                        empty_setup_guard.start, empty_elevated,
                    ):
                        raise RuntimeError(f'Empty pickup opening/torso rejected: {empty_setup_guard.last_rejection}')
                solutions, orientation_index, path_score, transition_waypoints = (
                    self._solve_cartesian_path(
                        positions,
                        rotations,
                        pick_torso_height,
                        endpoint_first=top_row,
                        position_tolerance=pick_position_tolerance,
                        orientation_tolerance=pick_orientation_tolerance,
                        **empty_setup_options,
                    )
                )
        # Two same-seed physics runs isolated top-row book/arm contact to the
        # final loaded extraction leg (solutions[1] -> solutions[0]).  The
        # preceding pose reduces the book/arm x-envelope overlap by about
        # 17 mm. With a shallower grasp, base retreat completes the shelf exit.
        # Keep solutions[0] in the unloaded
        # approach/recovery route, but plan and execute the loaded return from
        # solutions[1].  Other rows retain the fully retracted clearance pose.
        loaded_clearance_index = (lower_pick_plan.loaded_clearance_index
            if lower_pick_plan is not None
            else self._loaded_clearance_index(top_row, solutions))
        if loaded_clearance_index is None:
            if lower_pick_plan is None or not lower_pick_plan.extraction_solutions:
                raise RuntimeError('Separate lower withdrawal proposal is missing')
            loaded_clearance = lower_pick_plan.extraction_solutions[-1]
        else:
            if not 0 <= loaded_clearance_index < len(solutions) - 1:
                raise RuntimeError('Loaded clearance index is outside the solved pick path')
            loaded_clearance = solutions[loaded_clearance_index]
        accuracy = self._pick_path_accuracy(
            positions, rotations[orientation_index], solutions,
        )
        self._publish_status(
            'pick_approach_planned',
            command='pick',
            target=[float(value) for value in front],
            grasp=[float(value) for value in grasp],
            orientation_index=orientation_index,
            orientation=np.asarray(rotations[orientation_index]).tolist(),
            path_score=path_score,
            solutions=[np.asarray(solution).tolist() for solution in solutions],
            transition_solutions=[
                np.asarray(solution).tolist() for solution in transition_waypoints
            ],
            loaded_clearance_index=loaded_clearance_index,
            separate_withdrawal_proposal=(
                [q.tolist() for q in lower_pick_plan.extraction_solutions]
                if loaded_clearance_index is None else None),
            torso_height=pick_torso_height,
            grasp_depth_offset=depth_offset,
            pick_position_tolerance_m=pick_position_tolerance,
            pick_orientation_tolerance_rad=pick_orientation_tolerance,
            **accuracy,
        )
        if (
            accuracy['approach_max_position_error_m'] > pick_position_tolerance
            or accuracy['approach_max_orientation_error_rad'] > pick_orientation_tolerance
        ):
            raise RuntimeError(
                'Pick Cartesian alignment exceeds precision tolerance: '
                f"position={accuracy['approach_max_position_error_m']:.6f}m, "
                f"orientation={accuracy['approach_max_orientation_error_rad']:.6f}rad"
            )
        open_approach = check_open_gripper_approach(
            self, front, solutions, torso_height=pick_torso_height,
        )
        if not open_approach.ok:
            raise RuntimeError(
                f'Open gripper approach rejected: {open_approach.reason}; '
                f'link={open_approach.collision_link}, '
                f'leg={open_approach.leg_index}, sample={open_approach.sample_index}'
            )
        staged_empty_gripper = bool(
            top_row and getattr(self, 'shelf_side_cradle_enabled', False)
        )
        if staged_empty_gripper:
            from .shelf_cradle_geometry import (
                check_cradle_tool_route, check_gripper_opening,
            )
            setup_start = np.asarray(self._current_seed(), dtype=float).copy()
            setup_start[0] = pick_torso_height
            reason = check_cradle_tool_route(
                self, front, solutions[-1], setup_start,
                [*transition_waypoints, solutions[0]],
                float(front[0]) - 0.065, aperture=0.0,
            )
            if reason is not None:
                raise RuntimeError(f'Closed gripper setup rejected: {reason}')
            reason = check_gripper_opening(
                self, front, solutions[-1], solutions[0], float(front[0]) - 0.065,
            )
            if reason is not None:
                raise RuntimeError(f'Gripper opening at clearance rejected: {reason}')
            reason = check_cradle_tool_route(
                self, front, solutions[-1], solutions[0], solutions[1:],
                None, aperture=self.gripper_open,
            )
            if reason is not None:
                raise RuntimeError(f'Open gripper approach rejected: {reason}')
        with empty_torso_planning_scope(self, empty_setup_guard, lift_reference,
                ordinary=bool(top_row and lift_enabled and not staged_empty_gripper)) as torso_overlap:
            extraction_solutions = (
                list(lower_pick_plan.extraction_solutions)
                if loaded_clearance_index is None else
                list(reversed(solutions[loaded_clearance_index:-1])))
            post_retreat_result = None
            if lower_pick_plan is not None:
                lift_plan = lower_pick_plan.lift_plan
                extraction_solutions = list(lift_plan.route)
                loaded_clearance = lift_plan.terminal
                post_retreat_result = lower_pick_plan.return_result
                self._cached_post_retreat_plan = lower_pick_plan.cached_post_retreat_plan
            elif lift_enabled:
                from .lift_first_extraction import plan_lift_first_extraction
                self._lift_first_measurements(lift_reference)
                followup_options = {}
                if (getattr(self, 'pickup_post_retreat_parallel_geometry_enabled', False)
                        and top_row and not staged_empty_gripper):
                    def post_retreat_plan(planned_lift, geometry_backend):
                        options = {} if geometry_backend is None else {'geometry_backend': geometry_backend}
                        return self._plan_carried_return(front, solutions[-1], planned_lift.terminal,
                            rotations[orientation_index], pick_torso_height, **options)
                    followup_options['post_retreat_plan'] = post_retreat_plan
                lift_plan = run_pickup_geometry(
                    self, plan_lift_first_extraction, front, solutions[-1], extraction_solutions,
                    scene_reference=lift_reference, bay=lift_bay,
                    aperture=float(self.carried_book_dimensions[1]),
                    lift_m=self.lift_first_extraction_lift_m,
                    modeled_tool_allowance_m=.005,
                    **followup_options,
                )
                if followup_options:
                    lift_plan, post_retreat_result = lift_plan
                extraction_solutions = list(lift_plan.route)
                loaded_clearance = lift_plan.terminal
            (
                lowering,
                cradle_route,
                cradle_roll_index,
                transport_route,
                attached_corners,
            ) = (post_retreat_result if post_retreat_result is not None else self._plan_carried_return(
                front,
                solutions[-1],
                loaded_clearance,
                rotations[orientation_index],
                pick_torso_height,
            ))
            # ``_plan_carried_return`` preflights the exact top-row route that must
            # run immediately after the base retreat.  The following open is the
            # intentional *pre-grasp* reset, not a payload release, so preserve that
            # freshly audited plan across the otherwise state-clearing operation.
            planned_post_retreat_plan = self._cached_post_retreat_plan
            deferred_post_retreat_return = planned_post_retreat_plan is not None
            extraction_previous = solutions[-1]
            for solution in extraction_solutions:
                if not self._carried_robot_transition_is_safe(
                    extraction_previous,
                    solution,
                    attached_corners,
                ):
                    raise RuntimeError(
                        'Carried book would collide or rotate too far during extraction'
                    )
                extraction_previous = solution
            carried_terminal = np.asarray(
                (
                    transport_route[-1]
                    if transport_route
                    else cradle_route[-1]
                    if cradle_route
                    else lowering[-1]
                    if lowering
                    else loaded_clearance
                ),
                dtype=float,
            )
            unloaded_recovery_route: List[np.ndarray] = []
            if not transport_route:
                recovery_start = carried_terminal.copy()
                recovery_start[0] = HOME[0]
                recovery_transition = self._plan_retracted_transition(
                    recovery_start,
                    HOME,
                )
                if recovery_transition is None:
                    raise RuntimeError('No safe unloaded recovery route from carry pose')
                unloaded_recovery_route = [*recovery_transition, HOME.copy()]
            prepared_empty_gripper = bool(
                staged_empty_gripper and getattr(self, '_empty_arm_staged', False)
            )
            if prepared_empty_gripper:
                # Reacquisition may select a different clearance posture. Only the
                # closed transition and opening at the NEW clearance were checked;
                # opening here would leave that certificate. Recheck empty closure
                # after planning before preserving it through the setup motion.
                measured, _ = check_empty_stationary(self)
                if abs(measured['gripper_left_finger_joint']) > .004:
                    raise RuntimeError('staged pick gripper no longer measures closed empty')
            self._publish_status(
                'ik_ready',
                command='pick',
                target=[float(value) for value in front],
                grasp=[float(value) for value in grasp],
                waypoints=len(solutions),
                orientation_index=orientation_index,
                path_score=path_score,
                staged_transition=bool(transition_waypoints),
                transition_waypoints=len(transition_waypoints),
                endpoint_first=(lower_pick_plan.endpoint_first
                    if lower_pick_plan is not None else top_row),
                grasp_depth_offset=depth_offset,
                grasp_lateral_offset=grasp_lateral_offset,
                grasp_vertical_offset=grasp_vertical_offset,
                loaded_clearance_lift=loaded_clearance_lift,
                lowering_waypoints=len(lowering),
                cradle_waypoints=len(cradle_route),
                cradle_roll_index=cradle_roll_index,
                carried_transport_waypoints=len(transport_route),
                post_retreat_compaction_required=not bool(transport_route),
                loaded_clearance_index=loaded_clearance_index,
                extraction_waypoints=len(extraction_solutions),
                lift_first_extraction_enabled=lift_enabled,
            )
            if lift_enabled:
                self._lift_first_measurements(lift_reference)
                self._publish_status('lift_first_extraction_planned', command='pick',
                    route=[q.tolist() for q in extraction_solutions],
                    terminal=loaded_clearance.tolist(),
                    planned_finger_geometry_basis='nominal_official_urdf_mimics_at_book_width',
                    **{**dict(lift_plan.metrics), 'deferred_carry_regenerated': True})
        # From here the checked pick intentionally approaches/touches a book.
        # Until this point the staged hand remains protected during base motion,
        # camera reacquisition and planning as well as during initial setup.
        self._empty_arm_contact_guard = False
        if empty_setup_guard is not None:
            empty_setup_guard.require_fresh(
                empty_setup_guard.start if torso_overlap is None else empty_elevated,
                empty_setup_guard.initial_aperture if torso_overlap is None else empty_setup_guard.open_aperture)
            self._publish_status('empty_pickup_geometry_verified', command='pick',
                samples=empty_setup_guard.checked_samples, exact_cache_hits=empty_setup_guard.cache_hits,
                measured_right=empty_setup_guard.right.tolist(), measured_head=empty_setup_guard.head.tolist(),
                measured_start=empty_setup_guard.start.tolist(), open_aperture=empty_setup_guard.open_aperture,
                screen_scope='URDF visual box planner envelope; official physics unchanged',
                geometry_scope='sampled nominal tool and robot; existing engineering feedback tolerances')
        if torso_overlap is None:
            if not prepared_empty_gripper and not self._best_effort(self._open_gripper):
                raise RuntimeError('pick_recovery_failed')
            if staged_empty_gripper and not self._best_effort(
                lambda: self._command_gripper(0.0)
            ):
                raise RuntimeError('pick_recovery_failed')
            if empty_setup_guard is not None:
                empty_setup_guard.require_fresh(empty_setup_guard.start, empty_setup_guard.open_aperture)
            if not self._best_effort(
                lambda: self._move_torso(pick_torso_height, 2.5)
            ):
                raise RuntimeError('pick_recovery_failed')
            if empty_setup_guard is not None:
                empty_setup_guard.wait_for_endpoint(
                    empty_setup_guard.start, empty_elevated,
                    aperture=empty_setup_guard.open_aperture, phase='empty_setup_torso',
                )
        empty_expected = (empty_elevated if empty_setup_guard is not None else None)
        for waypoint in transition_waypoints:
            if empty_setup_guard is not None:
                empty_setup_guard.require_fresh(empty_expected, empty_setup_guard.open_aperture)
            if not self._best_effort(
                lambda q=waypoint: move_empty_pickup_setup(self, empty_setup_guard,
                    empty_expected, q, 2.2, phase='empty_setup_transition')
            ):
                raise RuntimeError('pick_recovery_failed')
            if empty_setup_guard is not None:
                empty_setup_guard.wait_for_endpoint(
                    empty_expected, waypoint, aperture=empty_setup_guard.open_aperture,
                    phase='empty_setup_transition',
                )
            empty_expected = waypoint
        if empty_setup_guard is not None:
            empty_setup_guard.require_fresh(empty_expected, empty_setup_guard.open_aperture)
        if not self._best_effort(
            lambda: move_empty_pickup_setup(self, empty_setup_guard,
                empty_expected, solutions[0], 2.8, phase='empty_setup_clearance')
        ):
            raise RuntimeError('pick_recovery_failed')
        if empty_setup_guard is not None:
            empty_setup_guard.wait_for_endpoint(
                empty_expected, solutions[0], aperture=empty_setup_guard.open_aperture,
                phase='empty_setup_clearance',
            )
        # The alternate IK branch passes near arm link 1 during retracted
        # setup. Keep the empty fingers folded for that checked route, then
        # open at Cartesian clearance before approaching the resting book.
        if staged_empty_gripper and not self._best_effort(self._open_gripper):
            raise RuntimeError('pick_recovery_failed')
        empty_expected = solutions[0]
        for solution in solutions[1:]:
            if empty_setup_guard is not None:
                empty_setup_guard.require_fresh(empty_expected, empty_setup_guard.open_aperture)
            if not self._best_effort(
                lambda q=solution: self._move_arm_solution(q, 0.65)
            ):
                raise RuntimeError('pick_recovery_failed')
            if empty_setup_guard is not None:
                empty_setup_guard.wait_for_endpoint(
                    empty_expected, solution, aperture=empty_setup_guard.open_aperture,
                    phase='empty_cartesian_approach',
                )
            empty_expected = solution
        if empty_setup_guard is not None:
            empty_setup_guard.require_fresh(empty_expected, empty_setup_guard.open_aperture)

        try:
            if interval_enabled:
                if not np.array_equal(attached_corners, lift_plan.attached_corners):
                    raise RuntimeError('preclose aperture attachment differs from carried route')
                prepare_interval(self, front, solutions[-1], lift_plan, lift_bay, lift_reference)
            close_result = self._close_for_grasp()
        except Exception:
            if interval_enabled:
                invalidate(self)
            return self._recover_ambiguous_grasp(
                attached_corners,
                solutions,
                transition_waypoints,
            )
        (
            grasp_verified,
            grasp_width,
            left_contact,
            right_contact,
            plausible_width,
        ) = close_result
        if not grasp_verified:
            if interval_enabled:
                invalidate(self)
            return self._recover_ambiguous_grasp(
                attached_corners,
                solutions,
                transition_waypoints,
            )

        # Install retained-payload state only after the guarded close has
        # proved that the planned rigid transform now exists.  From this point
        # onward the asynchronous monitor covers extraction and the long cradle
        # roll as well as navigation; before the roll it still requires a true
        # bilateral pinch.
        lock = getattr(self, '_lock', None)
        if lock is None:
            self._held_book_corners = attached_corners.copy()
            self._cached_post_retreat_plan = planned_post_retreat_plan
            self._gravity_supported_payload = False
            self._payload_hazard_latched = None
            self._payload_monitor_enabled = True
        else:
            with lock:
                self._held_book_corners = attached_corners.copy()
                self._cached_post_retreat_plan = planned_post_retreat_plan
                self._gravity_supported_payload = False
                self._payload_hazard_latched = None
                self._payload_monitor_enabled = True

        # Retry 23's 0.65 s / <=60 mm loaded legs outran the ordinary pinch
        # while the book was still sliding on its high-friction shelf.  Match
        # the roughly 10 mm/s rate already demonstrated by the guarded 5 mm
        # extraction diagnostic, without changing waypoint density or the
        # top-row loaded-clearance index.
        carried_legs: List[Tuple[Sequence[float], float, str]] = [
            (solution, 1.0 if lift_enabled and index == 0 else 5.8,
             'initial_shelf_lift' if lift_enabled and index == 0 else 'extraction')
            for index, solution in enumerate(extraction_solutions)
        ]
        carried_legs.extend(
            (solution, 0.65, 'fixed_orientation_lowering')
            for solution in lowering
        )
        transport_previous = np.asarray(
            lowering[-1] if lowering else loaded_clearance,
            dtype=float,
        )
        carried_roll_leg_index: Optional[int] = None
        for index, solution in enumerate(cradle_route):
            is_roll = index == cradle_roll_index
            if is_roll:
                carried_roll_leg_index = len(carried_legs)
            carried_legs.append(
                (
                    solution,
                    (
                        self.carried_cradle_transfer
                        if is_roll
                        else self._transport_leg_duration(
                            transport_previous,
                            solution,
                        )
                    ),
                    (
                        'mechanical_cradle'
                        if is_roll
                        else 'cradle_retraction'
                        if cradle_roll_index is not None
                        and index < cradle_roll_index
                        else 'supported_cradle_staging'
                    ),
                )
            )
            transport_previous = np.asarray(solution, dtype=float)
        for index, solution in enumerate(transport_route):
            carried_legs.append(
                (
                    solution,
                    self._transport_leg_duration(transport_previous, solution),
                    (
                        'transport_transition'
                        if index < len(transport_route) - 1
                        else 'transport'
                    ),
                )
            )
            transport_previous = np.asarray(solution, dtype=float)
        gravity_supported = carried_roll_leg_index is not None
        # Retain a strict bilateral pinch through extraction.  Only enable the
        # lower-finger retention rule immediately before the audited q7 roll;
        # the live book then rests on that jaw and is expected to lose contact
        # with the upper/right jaw.  Keeping the two phases separate prevents a
        # one-sided pre-roll slip from being mistaken for successful support.
        first_segment = (
            carried_legs[:carried_roll_leg_index]
            if carried_roll_leg_index is not None
            else carried_legs
        )
        if lift_enabled:
            try:
                if interval_enabled:
                    checked_hold = reuse_interval(
                        self, front, solutions[-1], lift_plan, lift_bay, lift_reference)
                else:
                    checked_hold = self._recheck_lift_first_geometry(
                        front, solutions[-1], lift_plan, lift_bay, lift_reference,
                    )
            except Exception as exc:
                # An invalid loaded transform cannot authorize either the
                # unchecked lift or a recovery route from its assumed endpoint.
                # Keep the acquired hand closed and stop for explicit recovery.
                self._publish_status('lift_first_preflight_rejected', command='pick',
                    reason=str(exc), retained_stop=True, recovery_halted=True,
                    **getattr(exc, 'master_geometry', {}))
                raise RuntimeError('pick_recovery_failed') from exc
            if not self._fresh_retention_probe('pick', 'before_initial_shelf_lift'):
                return self._recover_closed_pick(
                    lower_shelf_pick=not top_row,
                    cause='contact_lost', remaining_carried_legs=carried_legs,
                    unloaded_recovery_route=unloaded_recovery_route,
                )
            # Fresh contact may include a dwell; base/right/head must still be
            # the same measured collision context immediately before dispatch.
            try:
                current_hold = self._lift_first_measurements(lift_reference)
                if interval_enabled:
                    check_after_probe(self, checked_hold, current_hold)
                ManipulationNode._check_lift_collision_context(
                    current_hold, checked_hold.get('geometry_context'))
                if current_hold.get('finger_geometry_basis') != checked_hold.get('finger_geometry_basis'):
                    raise RuntimeError('lift-first finger feedback availability changed during contact probe')
                if np.max(np.abs(current_hold['left']-checked_hold['left'])) > .0002:
                    raise RuntimeError('lift-first arm moved during final contact probe')
                master_tolerance = (self.adaptive_endpoint_tolerance if _stock_diagnostic(self) else .00001)
                master_geometry = dict(
                    geometry_reference_master_position_m=checked_hold.get('geometry_reference_master_position_m', checked_hold['fingers']['gripper_left_finger_joint']),
                    master_comparison_reference_position_m=checked_hold['fingers']['gripper_left_finger_joint'],
                    current_master_position_m=current_hold['fingers']['gripper_left_finger_joint'],
                    master_geometry_tolerance_m=master_tolerance,
                    master_geometry_scope='nominal sampled geometry plus engineering admission tolerance; no geometry recomputation or passive deflection proof')
                for name, value in checked_hold['fingers'].items():
                    tolerance = master_tolerance if name == 'gripper_left_finger_joint' else .001
                    if abs(current_hold['fingers'][name]-value) > tolerance:
                        error = RuntimeError('lift-first finger moved during final contact probe')
                        error.master_geometry = master_geometry
                        raise error
            except Exception as exc:
                self._publish_status('lift_first_preflight_rejected', command='pick',
                    phase='after_contact_probe', reason=str(exc),
                    retained_stop=True, recovery_halted=True,
                    **getattr(exc, 'master_geometry', {}))
                raise RuntimeError('pick_recovery_failed') from exc
        initial_pressure_gate = (
            LiftPressureGate(self, checked_hold, lift_reference) if lift_enabled else None
        )
        # The optional timing certificate covers the ordinary top-row lift and
        # its four withdrawal legs. Lower shelves use different admitted paths
        # and retain their original durations and the same pressure/contact gates.
        ordinary_top_withdrawal = bool(
            top_row and lift_enabled and not staged_empty_gripper
            and deferred_post_retreat_return
        )
        try:
            path_ok, next_leg, contact_lost = self._execute_retained_arm_legs(
                first_segment,
                'pick',
                **({'fresh_retention_phases': ('initial_shelf_lift',)} if lift_enabled else {}),
                **({'initial_pressure_gate': initial_pressure_gate} if lift_enabled else {}),
                **({'withdrawal_speed_scale': self.withdrawal_speed_scale}
                   if ordinary_top_withdrawal and getattr(self, 'withdrawal_speed_scale', 1.0) != 1.0 else {}),
                **withdrawal_half_normal_options(self, first_segment, initial_pressure_gate,
                    lift_plan if lift_enabled else None,
                    ordinary=ordinary_top_withdrawal),
            )
        except LiftPressureRejected as exc:
            raise RuntimeError('pick_recovery_failed') from exc
        if not path_ok:
            # Whether contact was lost or an action aborted, clear the physical
            # held state before the mission manager is allowed to retry.  For an
            # failure or in-flight contact loss, ``next_leg`` is the interrupted
            # segment. Only contact loss found after a completed endpoint probe
            # advances it. Recovery must never assume a partial endpoint was
            # reached or replay a carried route after losing its rigid payload.
            return self._recover_closed_pick(
                lower_shelf_pick=not top_row,
                cause='contact_lost' if contact_lost else 'motion_failed',
                remaining_carried_legs=carried_legs[next_leg:],
                unloaded_recovery_route=unloaded_recovery_route,
            )
        if carried_roll_leg_index is not None:
            # Per-leg checks accept contacts up to grasp_contact_max_age old.
            # Clear those samples and prove a fresh bilateral pinch immediately
            # before changing to the one-sided, gravity-supported criterion.
            if not self._fresh_retention_probe(
                'pick',
                'pre_cradle',
                leg=max(0, carried_roll_leg_index - 1),
            ):
                return self._recover_closed_pick(
                    lower_shelf_pick=not top_row,
                    cause='contact_lost',
                    remaining_carried_legs=carried_legs[
                        carried_roll_leg_index:
                    ],
                    unloaded_recovery_route=unloaded_recovery_route,
                )
            roll_solution, roll_duration, roll_phase = carried_legs[
                carried_roll_leg_index
            ]
            # During the q7 sweep the contact mode changes continuously from a
            # bilateral pinch to lower-jaw support.  Suppress the generic
            # fixed-mode timer for this one audited transition; otherwise the
            # expected upper-jaw separation would be misreported as a drop.
            # Arm the robot-contact watchdog and conservative supported-state
            # recovery before dispatch.  A rejected or partially completed
            # controller goal must be cancelled on collision and must never
            # open the gripper from an unknown pose above the arm.
            lock = getattr(self, '_lock', None)
            if lock is None:
                self._retention_probe_active = True
                self._gravity_supported_payload = True
                self._payload_robot_watchdog_enabled = True
            else:
                with lock:
                    self._retention_probe_active = True
                    self._gravity_supported_payload = True
                    self._payload_robot_watchdog_enabled = True
            roll_moved = False
            try:
                try:
                    roll_moved = self._move_arm_solution(
                        roll_solution,
                        roll_duration,
                    )
                except Exception as exc:
                    self._publish_status(
                        'motion_exception',
                        command='pick',
                        phase=roll_phase,
                        leg=carried_roll_leg_index,
                        reason=str(exc),
                    )
            finally:
                if lock is None:
                    self._retention_probe_active = False
                    self._payload_robot_watchdog_enabled = False
                else:
                    with lock:
                        self._retention_probe_active = False
                        self._payload_robot_watchdog_enabled = False
            if not roll_moved:
                return self._recover_closed_pick(
                    lower_shelf_pick=not top_row,
                    cause='motion_failed',
                    remaining_carried_legs=carried_legs[
                        carried_roll_leg_index:
                    ],
                    unloaded_recovery_route=unloaded_recovery_route,
                )
            # The controller reached the exact audited support attitude.  Only
            # now may a fresh contact on the lower/left jaw replace bilateral
            # pinch as the retained-payload criterion.
            if not self._retention_after_leg(
                'pick',
                roll_phase,
                carried_roll_leg_index,
            ):
                return self._recover_closed_pick(
                    lower_shelf_pick=not top_row,
                    cause='contact_lost',
                    remaining_carried_legs=carried_legs[
                        carried_roll_leg_index + 1:
                    ],
                    unloaded_recovery_route=unloaded_recovery_route,
                )
            # A normal per-leg check can still accept a contact callback that
            # predates the controlled roll.  Clear both timestamps at the roll
            # endpoint, then require newly observed lower-finger support before
            # allowing the base to perform the straight shelf-clear retreat.
            if not self._fresh_retention_probe(
                'pick',
                'mechanical_cradle',
                leg=carried_roll_leg_index,
            ):
                return self._recover_closed_pick(
                    lower_shelf_pick=not top_row,
                    cause='cradle_contact_verification_failed',
                    remaining_carried_legs=carried_legs[
                        carried_roll_leg_index + 1:
                    ],
                    unloaded_recovery_route=unloaded_recovery_route,
                )
            remaining_segment = carried_legs[carried_roll_leg_index + 1:]
            path_ok, next_leg, contact_lost = (
                self._execute_retained_arm_legs(
                    remaining_segment,
                    'pick',
                    fresh_retention_phases=('supported_cradle_staging',),
                    leg_offset=carried_roll_leg_index + 1,
                )
            )
            if not path_ok:
                return self._recover_closed_pick(
                    lower_shelf_pick=not top_row,
                    cause=(
                        'contact_lost' if contact_lost else 'motion_failed'
                    ),
                    remaining_carried_legs=remaining_segment[next_leg:],
                    unloaded_recovery_route=unloaded_recovery_route,
                )
        # Keep a top-row payload and torso unchanged in its mechanical cradle
        # for the straight base retreat.  Lower rows retain the compact-at-shelf
        # torso return that was already validated in simulation.
        if not deferred_post_retreat_return:
            if not self._best_effort(
                lambda: self._move_torso(float(HOME[0]), 2.0)
            ):
                return self._recover_closed_pick(
                    lower_shelf_pick=not top_row,
                    cause='torso_motion_failed',
                    remaining_carried_legs=(),
                    unloaded_recovery_route=unloaded_recovery_route,
                )
            if not self._retention_after_leg('pick', 'torso_home', 0):
                return self._recover_closed_pick(
                    lower_shelf_pick=not top_row,
                    cause='contact_lost_after_torso',
                    remaining_carried_legs=(),
                    unloaded_recovery_route=unloaded_recovery_route,
                )
        # Do not let the contact samples that completed the final trajectory
        # immediately satisfy the success gate.  The failed seed-101 cradle
        # detached 0.31 s after that gate.  Clear both timestamps, dwell for a
        # full gripper-settle interval, and require a newly observed contact in
        # the active retention mode before allowing the base to move.
        # Clearing a healthy contact epoch intentionally creates a brief empty
        # snapshot.  Suppress the asynchronous 50 ms watchdog until the final
        # sample is read, just as ``_fresh_retention_probe`` does; otherwise the
        # watchdog can publish a false loss in the clear-to-callback window.
        lock = getattr(self, '_lock', None)
        if lock is None:
            self._retention_probe_active = True
        else:
            with lock:
                self._retention_probe_active = True
        try:
            self._clear_target_contact_samples()
            settled = self._wait_sim_duration(self.gripper_settle)
            contact_probe_seconds = min(0.20, 0.5 * self.grasp_contact_max_age)
            if settled:
                # Samples received during the dwell prove only that contact
                # existed at some point in that interval.  Clear them again and
                # require a new sample after the full dwell has elapsed.
                self._clear_target_contact_samples()
                settled = self._wait_sim_duration(contact_probe_seconds)
            (
                retained,
                grasp_width,
                left_contact,
                right_contact,
                plausible_width,
            ) = self._pinch_sample(
                max_age=min(0.15, float(self.grasp_contact_max_age)),
            )
        finally:
            if lock is None:
                self._retention_probe_active = False
            else:
                with lock:
                    self._retention_probe_active = False
        carried_staging_solution = carried_terminal.copy()
        if not deferred_post_retreat_return:
            carried_staging_solution[0] = HOME[0]
        post_retreat_shelf_front_x = (
            float(planned_post_retreat_plan['shelf_front_x'])
            if planned_post_retreat_plan is not None
            and 'shelf_front_x' in planned_post_retreat_plan
            else float(front[0]) + self.carried_shelf_retreat_clearance
            if not transport_route
            else None
        )

        def commit_success_state() -> Optional[str]:
            payload_hazard = getattr(self, '_payload_hazard_latched', None)
            if getattr(self, '_target_robot_contact_latched', False):
                payload_hazard = 'payload_robot_contact'
            if (
                payload_hazard is None
                and settled
                and retained
                and plausible_width
            ):
                self._carried_staging_solution = carried_staging_solution
                self._post_retreat_shelf_front_x = post_retreat_shelf_front_x
                self._gravity_supported_payload = gravity_supported
                self._supported_post_retreat_staging_required = (
                    deferred_post_retreat_return
                )
                self._payload_monitor_enabled = True
            return payload_hazard

        if lock is None:
            payload_hazard = commit_success_state()
        else:
            with lock:
                payload_hazard = commit_success_state()
        retained = bool(retained and payload_hazard is None)
        self._publish_status(
            'grasp_verified',
            command='pick',
            retained=retained,
            measured_position=grasp_width,
            left_contact=left_contact,
            right_contact=right_contact,
            bilateral_contact=left_contact and right_contact,
            plausible_width=plausible_width,
            settled=settled,
            contact_probe_seconds=contact_probe_seconds,
            payload_hazard=payload_hazard,
        )
        if not settled or not retained or not plausible_width or payload_hazard:
            if payload_hazard is not None:
                recovery_cause = payload_hazard
            elif not settled:
                recovery_cause = 'final_grasp_settle_failed'
            elif not retained:
                recovery_cause = 'final_grasp_contact_lost'
            else:
                recovery_cause = 'final_grasp_width_invalid'
            return self._recover_closed_pick(
                lower_shelf_pick=not top_row,
                cause=recovery_cause,
                remaining_carried_legs=(),
                unloaded_recovery_route=unloaded_recovery_route,
            )
        return True

    def _execute_cached_post_retreat_compaction(
        self,
        current: Sequence[float],
        shelf_front_x: Optional[float],
    ) -> bool:
        """Execute vertical extension, cradle roll, then supported compaction."""
        cached_plan = getattr(self, '_cached_post_retreat_plan', None)
        if not isinstance(cached_plan, dict):
            raise RuntimeError('cached post-retreat plan is unavailable')
        if (
            cached_plan.get('requires_gravity_support') is not False
            or bool(getattr(self, '_gravity_supported_payload', False))
        ):
            raise RuntimeError(
                'cached post-retreat plan requires a vertical bilateral pinch'
            )

        measured_start = np.asarray(current, dtype=float)
        try:
            cached_start = np.asarray(cached_plan.get('start'), dtype=float)
            cached_shelf_front_x = float(cached_plan.get('shelf_front_x'))
            cached_corners = np.asarray(
                cached_plan.get('attached_corners'), dtype=float
            )
            raw_legs = list(cached_plan.get('legs', ()))
            staging_terminal = np.asarray(
                cached_plan.get('staging_terminal'), dtype=float
            ).copy()
            cached_terminal = np.asarray(
                cached_plan.get('terminal'), dtype=float
            ).copy()
            compact_radius = float(cached_plan.get('compact_radius'))
        except (TypeError, ValueError):
            raise RuntimeError('cached post-retreat plan is malformed') from None

        if (
            cached_start.shape != (8,)
            or not np.all(np.isfinite(cached_start))
            or shelf_front_x is None
            or not np.isfinite(cached_shelf_front_x)
            or abs(cached_shelf_front_x - shelf_front_x) > 1e-6
            or cached_corners.shape != (8, 3)
            or not np.allclose(
                cached_corners,
                self._held_book_corners,
                rtol=0.0,
                atol=1e-12,
            )
            or staging_terminal.shape != (8,)
            or not np.all(np.isfinite(staging_terminal))
            or abs(float(staging_terminal[0] - cached_start[0])) > 1e-12
            or cached_terminal.shape != (8,)
            or not np.all(np.isfinite(cached_terminal))
            or abs(float(cached_terminal[0] - cached_start[0])) > 1e-12
            or not np.isfinite(compact_radius)
            or compact_radius > self.carried_navigation_radius_limit
            or abs(float(measured_start[0] - cached_start[0])) > 0.001
            or float(
                np.max(np.abs(measured_start[1:] - cached_start[1:]))
            ) > 0.005
            or not raw_legs
        ):
            raise RuntimeError('cached post-retreat plan does not match measured state')

        allowed_phases = (
            'post_retreat_clearance_extension',
            'post_retreat_cradle_roll',
            'supported_cradle_lowering',
            'supported_cradle_retraction',
            'compact_transport',
        )
        phase_rank = {
            phase: index for index, phase in enumerate(allowed_phases)
        }
        parsed: List[Tuple[np.ndarray, str]] = []
        previous_rank = -1
        for raw_leg in raw_legs:
            if not isinstance(raw_leg, (tuple, list)) or len(raw_leg) != 2:
                raise RuntimeError('cached post-retreat plan is malformed')
            raw_solution, raw_phase = raw_leg
            solution = np.asarray(raw_solution, dtype=float).copy()
            phase = str(raw_phase)
            rank = phase_rank.get(phase, -1)
            if (
                solution.shape != (8,)
                or not np.all(np.isfinite(solution))
                or abs(float(solution[0] - cached_start[0])) > 1e-12
                or rank < previous_rank
                or rank < 0
            ):
                raise RuntimeError('cached post-retreat plan is malformed')
            parsed.append((solution, phase))
            previous_rank = rank

        phases = [phase for _, phase in parsed]
        phase_counts = {
            phase: phases.count(phase) for phase in allowed_phases
        }
        if any(phase_counts[phase] <= 0 for phase in allowed_phases):
            raise RuntimeError('cached post-retreat plan is malformed')
        retraction_terminal = parsed[
            sum(
                phase_counts[phase]
                for phase in allowed_phases[:4]
            )
            - 1
        ][0]
        if (
            not np.allclose(
                retraction_terminal,
                staging_terminal,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                parsed[-1][0],
                cached_terminal,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise RuntimeError('cached post-retreat plan is malformed')

        timed_groups: List[List[Tuple[np.ndarray, float, str]]] = [[], [], []]
        previous = cached_start.copy()
        for solution, phase in parsed:
            if phase == 'post_retreat_clearance_extension':
                group_index = 0
            elif phase == 'post_retreat_cradle_roll':
                group_index = 1
            else:
                group_index = 2
            duration = (
                self.carried_cradle_transfer
                / phase_counts['post_retreat_cradle_roll']
                if group_index == 1
                else self._transport_leg_duration(previous, solution)
            )
            timed_groups[group_index].append((solution, duration, phase))
            previous = solution

        first_solution = timed_groups[0][0][0]
        if not (
            self._carried_post_retreat_transition_is_safe(
                measured_start,
                measured_start,
                self._held_book_corners,
                cached_shelf_front_x,
            )
            and self._carried_post_retreat_transition_is_safe(
                measured_start,
                first_solution,
                self._held_book_corners,
                cached_shelf_front_x,
            )
        ):
            raise RuntimeError('measured cached start is not payload-safe')

        # Construct all three continuous action goals and discover the server
        # before the final physical proof.  This keeps the gap from proof to
        # motion constant-time while retaining a deliberate mode boundary at
        # the cradle roll.
        supported_watchdog_duration = None
        arm_speed_scale = getattr(self, 'placement_transport_speed_scale', 1.0)
        if arm_speed_scale != 1.0:
            arm_speed_scale = checked_arm_speed_scale(arm_speed_scale)
            supported_watchdog_duration = sum(item[1] for item in timed_groups[2])
            # Only the post-roll supported group uses the optional floor.
            supported_minimum = getattr(self, 'supported_compact_minimum_segment_seconds', .35)
            if not .175 <= supported_minimum <= .35:
                raise ValueError('supported_compact_minimum_segment_seconds must be within [.175, .35]')
            timed_groups[2] = [(q, scaled_arm_seconds(seconds, arm_speed_scale, minimum=supported_minimum), phase)
                               for q, seconds, phase in timed_groups[2]]
        prepared_groups = [
            self._make_retained_arm_trajectory_goal(group)
            for group in timed_groups
        ]
        if not self.arm_client.wait_for_server(
            timeout_sec=min(5.0, self.timeout)
        ):
            raise RuntimeError('left arm action server unavailable')
        if not self._fresh_retention_probe(
            'compact_transport',
            'post_retreat',
            require_new_sample=False,
        ):
            raise RuntimeError(
                'target-book contact was not retained before compaction'
            )
        dispatch_start = self._measured_left_solution()
        if (
            abs(float(dispatch_start[0] - cached_start[0])) > 0.001
            or float(
                np.max(np.abs(dispatch_start[1:] - cached_start[1:]))
            ) > 0.005
            or abs(float(dispatch_start[0] - measured_start[0])) > 0.0001
            or float(
                np.max(np.abs(dispatch_start[1:] - measured_start[1:]))
            ) > 0.0005
        ):
            raise RuntimeError('cached post-retreat plan drifted before dispatch')

        extension_goal, extension_duration = prepared_groups[0]
        extension_half_enabled = getattr(self, 'compact_extension_half_timing_enabled', False)
        if extension_half_enabled and self.additional_arm_time_scale != 2.0:
            raise ValueError('compact extension half timing requires additional_arm_time_scale=2')
        succeeded, _ = self._send_retained_arm_trajectory(
            extension_goal,
            extension_duration,
            timed_groups[0],
            'compact_transport',
            **({'velocity_admission': True, 'velocity_headroom': True}
               if extension_half_enabled else {}),
        )
        if not succeeded:
            raise RuntimeError('compact_transport_failed')
        expected_extension_terminal = timed_groups[0][-1][0]
        extension_terminal = self._wait_for_retained_endpoint(
            expected_extension_terminal,
            command='compact_transport',
            phase='post_retreat_clearance_extension',
            leg=len(timed_groups[0]) - 1,
        )
        if extension_terminal is None:
            raise RuntimeError(
                'post-retreat clearance extension missed its terminal state'
            )
        if not self._fresh_retention_probe(
            'compact_transport',
            'post_retreat_clearance_extension',
            leg=len(timed_groups[0]) - 1,
        ):
            raise RuntimeError('compact_transport_failed')
        extension_dispatch = self._measured_left_solution()
        if (
            abs(float(extension_dispatch[0] - expected_extension_terminal[0]))
            > 0.001
            or float(
                np.max(
                    np.abs(
                        extension_dispatch[1:]
                        - expected_extension_terminal[1:]
                    )
                )
            )
            > 0.005
        ):
            raise RuntimeError(
                'post-retreat clearance extension drifted before cradle roll'
            )

        # q7 changes the valid contact criterion continuously.  Suppress the
        # fixed-mode timer, enable lower-jaw support and the robot-contact
        # watchdog immediately before dispatch, and keep the supported state
        # fail-closed if the controller stops part-way through the roll.
        lock = getattr(self, '_lock', None)
        if lock is None:
            self._retention_probe_active = True
            self._gravity_supported_payload = True
            self._payload_robot_watchdog_enabled = True
        else:
            with lock:
                self._retention_probe_active = True
                self._gravity_supported_payload = True
                self._payload_robot_watchdog_enabled = True
        cradle_goal, cradle_duration = prepared_groups[1]
        cradle_terminal: Optional[np.ndarray] = None
        try:
            succeeded, _ = self._send_retained_arm_trajectory(
                cradle_goal,
                cradle_duration,
                timed_groups[1],
                'compact_transport',
            )
            if succeeded:
                cradle_terminal = self._wait_for_retained_endpoint(
                    timed_groups[1][-1][0],
                    command='compact_transport',
                    phase='post_retreat_cradle_roll',
                    leg=(
                        len(timed_groups[0])
                        + len(timed_groups[1])
                        - 1
                    ),
                )
        finally:
            if lock is None:
                self._retention_probe_active = False
                self._payload_robot_watchdog_enabled = False
            else:
                with lock:
                    self._retention_probe_active = False
                    self._payload_robot_watchdog_enabled = False
        if not succeeded:
            raise RuntimeError('compact_transport_failed')
        expected_cradle_terminal = timed_groups[1][-1][0]
        if cradle_terminal is None:
            raise RuntimeError('post-retreat cradle roll missed its terminal state')
        if not self._retention_after_leg(
            'compact_transport',
            'post_retreat_cradle_roll',
            len(timed_groups[0]) + len(timed_groups[1]) - 1,
        ):
            raise RuntimeError('compact_transport_failed')
        if not self._fresh_retention_probe(
            'compact_transport',
            'post_retreat_cradle_roll',
            leg=len(timed_groups[0]) + len(timed_groups[1]) - 1,
        ):
            raise RuntimeError('compact_transport_failed')
        supported_dispatch = self._measured_left_solution()
        if (
            abs(float(supported_dispatch[0] - expected_cradle_terminal[0]))
            > 0.001
            or float(
                np.max(
                    np.abs(
                        supported_dispatch[1:] - expected_cradle_terminal[1:]
                    )
                )
            )
            > 0.005
        ):
            raise RuntimeError(
                'post-retreat cradle roll drifted before supported compaction'
            )

        supported_goal, supported_duration = prepared_groups[2]
        succeeded, _ = self._send_retained_arm_trajectory(
            supported_goal,
            (supported_duration if supported_watchdog_duration is None
             else supported_watchdog_duration),
            timed_groups[2],
            'compact_transport',
            **({'velocity_admission': True} if arm_speed_scale != 1.0 else {}),
        )
        if not succeeded:
            raise RuntimeError('compact_transport_failed')

        measured_terminal = self._wait_for_retained_endpoint(
            cached_terminal,
            command='compact_transport',
            phase='compact_transport_final',
            leg=len(parsed) - 1,
        )
        if measured_terminal is None:
            raise RuntimeError(
                'cached post-retreat trajectory missed its terminal state'
            )
        if not (
            self._carried_post_retreat_transition_is_safe(
                measured_terminal,
                measured_terminal,
                self._held_book_corners,
                cached_shelf_front_x,
            )
            and self._gravity_supported_transition_is_safe(
                measured_terminal, measured_terminal
            )
        ):
            raise RuntimeError('Measured compact transport pose is not payload-safe')
        measured_radius = self._carried_navigation_radius(
            measured_terminal, self._held_book_corners
        )
        if measured_radius > self.carried_navigation_radius_limit:
            raise RuntimeError(
                'Measured compact transport exceeds navigation radius: '
                f'{measured_radius:.3f} > '
                f'{self.carried_navigation_radius_limit:.3f} m'
            )
        if not self._fresh_retention_probe(
            'compact_transport',
            'compact_transport_final',
            leg=len(parsed),
        ):
            raise RuntimeError('compact_transport_failed')

        self._carried_staging_solution = staging_terminal
        self._cached_post_retreat_plan = None
        self._post_retreat_shelf_front_x = None
        self._supported_post_retreat_staging_required = False
        self._publish_status(
            'transport_compact',
            command='compact_transport',
            waypoints=len(parsed),
            planar_radius=measured_radius,
        )
        return True

    def _compact_transport(self) -> bool:
        """Tuck a retained payload only after the base has cleared the shelf."""
        if self._held_book_corners is None:
            raise RuntimeError('compact transport requested without a book envelope')
        current = self._measured_left_solution()
        shelf_front_x = getattr(self, '_post_retreat_shelf_front_x', None)
        gravity_supported = bool(
            getattr(self, '_gravity_supported_payload', False)
        )
        staging_required = bool(
            getattr(self, '_supported_post_retreat_staging_required', False)
        )
        if staging_required and not gravity_supported:
            return self._execute_cached_post_retreat_compaction(
                current,
                shelf_front_x,
            )
        cached_deferred_route = staging_required
        if not cached_deferred_route and not self._fresh_retention_probe(
            'compact_transport',
            'post_retreat',
        ):
            raise RuntimeError(
                'target-book contact was not retained before compaction'
            )
        current_radius: Optional[float] = None
        # The deferred post-retreat route was densely collision-checked before
        # the pick.  Repeating even a single 61-sample mesh sweep here consumed
        # 2.65 simulated seconds in trial 749b26fe6bdd; a cached route instead
        # uses a strict, cheap measured-state identity gate below.
        if not cached_deferred_route:
            if shelf_front_x is None:
                current_safe = self._carried_robot_transition_is_safe(
                    current,
                    current,
                    self._held_book_corners,
                )
            else:
                current_safe = self._carried_post_retreat_transition_is_safe(
                    current,
                    current,
                    self._held_book_corners,
                    shelf_front_x,
                )
            if gravity_supported:
                current_safe = current_safe and (
                    self._gravity_supported_transition_is_safe(current, current)
                )
            if not current_safe:
                raise RuntimeError('Current carried pose is not payload-safe')
            current_radius = self._carried_navigation_radius(
                current,
                self._held_book_corners,
            )
        if (
            current_radius is not None
            and current_radius <= self.carried_navigation_radius_limit
            and not staging_required
        ):
            if not self._fresh_retention_probe(
                'compact_transport',
                'compact_transport_ready',
            ):
                raise RuntimeError('compact_transport_failed')
            self._cached_post_retreat_plan = None
            self._post_retreat_shelf_front_x = None
            self._supported_post_retreat_staging_required = False
            self._publish_status(
                'transport_compact',
                command='compact_transport',
                waypoints=0,
                planar_radius=current_radius,
            )
            return True

        route_kwargs = (
            {'post_retreat_shelf_front_x': shelf_front_x}
            if shelf_front_x is not None
            else {}
        )
        staging_terminal: Optional[np.ndarray] = None
        legs: List[Tuple[np.ndarray, float, str]] = []
        fresh_retention_phases: Tuple[str, ...] = ()
        cached_plan = getattr(self, '_cached_post_retreat_plan', None)
        if cached_deferred_route:
            if not isinstance(cached_plan, dict):
                raise RuntimeError('cached post-retreat plan is unavailable')
            if cached_plan.get('requires_gravity_support') is False:
                raise RuntimeError(
                    'cached post-retreat plan requires a vertical bilateral pinch'
                )
            if (
                cached_plan.get('requires_gravity_support') is not True
                or not gravity_supported
            ):
                raise RuntimeError(
                    'cached post-retreat plan requires gravity-supported payload state'
                )
            try:
                cached_start = np.asarray(cached_plan.get('start'), dtype=float)
                cached_shelf_front_x = float(cached_plan.get('shelf_front_x'))
                cached_corners = np.asarray(
                    cached_plan.get('attached_corners'),
                    dtype=float,
                )
                cached_legs = list(cached_plan.get('legs', ()))
                staging_terminal = np.asarray(
                    cached_plan.get('staging_terminal'),
                    dtype=float,
                ).copy()
                cached_terminal = np.asarray(
                    cached_plan.get('terminal'),
                    dtype=float,
                ).copy()
                compact_radius = float(cached_plan.get('compact_radius'))
            except (TypeError, ValueError):
                raise RuntimeError(
                    'cached post-retreat plan is malformed'
                ) from None
            if (
                cached_start.shape != (8,)
                or not np.all(np.isfinite(cached_start))
                or shelf_front_x is None
                or not np.isfinite(cached_shelf_front_x)
                or abs(cached_shelf_front_x - shelf_front_x) > 1e-6
                or cached_corners.shape != (8, 3)
                or not np.allclose(
                    cached_corners,
                    self._held_book_corners,
                    rtol=0.0,
                    atol=1e-12,
                )
                or staging_terminal.shape != (8,)
                or not np.all(np.isfinite(staging_terminal))
                or abs(float(staging_terminal[0] - cached_start[0])) > 1e-12
                or cached_terminal.shape != (8,)
                or not np.all(np.isfinite(cached_terminal))
                or abs(float(cached_terminal[0] - cached_start[0])) > 1e-12
                or not np.isfinite(compact_radius)
                or abs(float(current[0] - cached_start[0])) > 0.001
                or float(np.max(np.abs(current[1:] - cached_start[1:]))) > 0.005
                or not cached_legs
            ):
                raise RuntimeError('cached post-retreat plan does not match measured state')
            previous = np.asarray(current, dtype=float)
            post_retreat_roll_legs = sum(
                1
                for cached_leg in cached_legs
                if isinstance(cached_leg, (tuple, list))
                and len(cached_leg) == 2
                and str(cached_leg[1]) == 'post_retreat_cradle_roll'
            )
            for cached_leg in cached_legs:
                if not isinstance(cached_leg, (tuple, list)) or len(cached_leg) != 2:
                    raise RuntimeError('cached post-retreat plan is malformed')
                solution, phase = cached_leg
                copied = np.asarray(solution, dtype=float).copy()
                if (
                    copied.shape != (8,)
                    or not np.all(np.isfinite(copied))
                    or abs(float(copied[0] - cached_start[0])) > 1e-12
                ):
                    raise RuntimeError('cached post-retreat plan is malformed')
                legs.append(
                    (
                        copied,
                        (
                            self.carried_cradle_transfer
                            / post_retreat_roll_legs
                            if str(phase) == 'post_retreat_cradle_roll'
                            and post_retreat_roll_legs > 0
                            else self._transport_leg_duration(previous, copied)
                        ),
                        str(phase),
                    )
                )
                previous = copied
            if not np.allclose(
                previous,
                cached_terminal,
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError('cached post-retreat plan is malformed')
            first_cached_solution = legs[0][0]
            if not (
                self._carried_post_retreat_transition_is_safe(
                    current,
                    current,
                    self._held_book_corners,
                    cached_shelf_front_x,
                )
                and self._gravity_supported_transition_is_safe(current, current)
                and self._carried_post_retreat_transition_is_safe(
                    current,
                    first_cached_solution,
                    self._held_book_corners,
                    cached_shelf_front_x,
                )
                and self._gravity_supported_transition_is_safe(
                    current,
                    first_cached_solution,
                )
            ):
                raise RuntimeError(
                    'measured cached start is not payload-safe'
                )
        else:
            staged_legs: List[Tuple[np.ndarray, str]] = []
            compact_start = np.asarray(current, dtype=float)
            compact_goal = (
                SUPPORTED_CARRY.copy()
                if gravity_supported
                else CARRY.copy()
            )
            # Compact transport commands the arm controller only.  Keep the
            # measured torso coordinate fixed so planned collision geometry
            # matches every waypoint that the controller will actually execute.
            compact_goal[0] = compact_start[0]
            compact_goals = (
                self._supported_compact_goals(compact_start)
                if gravity_supported
                else [compact_goal]
            )
            route = self._plan_carried_joint_route(
                compact_start,
                compact_goals,
                self._held_book_corners,
                require_gravity_support=gravity_supported,
                **route_kwargs,
            )
            if route is None:
                raise RuntimeError('No payload-safe compact transport route')
            compact_radius = self._carried_navigation_radius(
                route[-1],
                self._held_book_corners,
            )
            previous = np.asarray(current, dtype=float)
            for solution, phase in (
                *staged_legs,
                *((solution, 'compact_transport') for solution in route),
            ):
                copied = np.asarray(solution, dtype=float)
                legs.append(
                    (
                        copied,
                        self._transport_leg_duration(previous, copied),
                        phase,
                    )
                )
                previous = copied
            if not self._fresh_retention_probe(
                'compact_transport',
                'pre_compact_motion',
            ):
                raise RuntimeError('compact_transport_failed')
        if compact_radius > self.carried_navigation_radius_limit:
            raise RuntimeError(
                'Compact transport exceeds navigation radius: '
                f'{compact_radius:.3f} > {self.carried_navigation_radius_limit:.3f} m'
            )
        if cached_deferred_route:
            # Build the complete multi-point goal and discover the server before
            # the final physical proof.  After that proof, only constant-time
            # gates remain before send_goal_async; this prevents the stationary
            # gravity-creep window seen in trial 749b26fe6bdd.  One controller
            # goal also removes 31 stop/start handshakes from the cached route.
            retained_goal, retained_duration = (
                self._make_retained_arm_trajectory_goal(legs)
            )
            if not self.arm_client.wait_for_server(
                timeout_sec=min(5.0, self.timeout)
            ):
                raise RuntimeError('left arm action server unavailable')
            if not self._fresh_retention_probe(
                'compact_transport',
                'post_retreat',
                require_new_sample=False,
            ):
                raise RuntimeError(
                    'target-book contact was not retained before compaction'
                )
            # Recheck after action-server discovery and the contact probe.  A
            # controller or external disturbance must not move the arm away
            # from the cached start before the first preflighted segment.
            dispatch_start = self._measured_left_solution()
            if (
                abs(float(dispatch_start[0] - cached_start[0])) > 0.001
                or float(
                    np.max(np.abs(dispatch_start[1:] - cached_start[1:]))
                ) > 0.005
                or abs(float(dispatch_start[0] - current[0])) > 0.0001
                or float(
                    np.max(np.abs(dispatch_start[1:] - current[1:]))
                ) > 0.0005
            ):
                raise RuntimeError(
                    'cached post-retreat plan drifted before dispatch'
                )
            succeeded, contact_lost = self._send_retained_arm_trajectory(
                retained_goal,
                retained_duration,
                legs,
                'compact_transport',
            )
        else:
            succeeded, _, contact_lost = self._execute_retained_arm_legs(
                legs,
                'compact_transport',
                fresh_retention_phases=fresh_retention_phases,
            )
        if not succeeded:
            raise RuntimeError('compact_transport_failed')

        measured_terminal = self._measured_left_solution()
        if cached_deferred_route:
            if (
                abs(float(measured_terminal[0] - cached_terminal[0])) > 0.001
                or float(
                    np.max(np.abs(measured_terminal[1:] - cached_terminal[1:]))
                ) > 0.005
            ):
                raise RuntimeError(
                    'cached post-retreat trajectory missed its terminal state'
                )
            measured_safe = (
                self._carried_post_retreat_transition_is_safe(
                    measured_terminal,
                    measured_terminal,
                    self._held_book_corners,
                    cached_shelf_front_x,
                )
                and self._gravity_supported_transition_is_safe(
                    measured_terminal,
                    measured_terminal,
                )
            )
            if not measured_safe:
                raise RuntimeError(
                    'Measured compact transport pose is not payload-safe'
                )
            compact_radius = self._carried_navigation_radius(
                measured_terminal,
                self._held_book_corners,
            )
            if compact_radius > self.carried_navigation_radius_limit:
                raise RuntimeError(
                    'Measured compact transport exceeds navigation radius: '
                    f'{compact_radius:.3f} > '
                    f'{self.carried_navigation_radius_limit:.3f} m'
                )
        else:
            if shelf_front_x is None:
                measured_safe = self._carried_robot_transition_is_safe(
                    measured_terminal,
                    measured_terminal,
                    self._held_book_corners,
                )
            else:
                measured_safe = self._carried_post_retreat_transition_is_safe(
                    measured_terminal,
                    measured_terminal,
                    self._held_book_corners,
                    shelf_front_x,
                )
            if gravity_supported:
                measured_safe = measured_safe and (
                    self._gravity_supported_transition_is_safe(
                        measured_terminal,
                        measured_terminal,
                    )
                )
            if not measured_safe:
                raise RuntimeError(
                    'Measured compact transport pose is not payload-safe'
                )
            compact_radius = self._carried_navigation_radius(
                measured_terminal,
                self._held_book_corners,
            )
            if compact_radius > self.carried_navigation_radius_limit:
                raise RuntimeError(
                    'Measured compact transport exceeds navigation radius: '
                    f'{compact_radius:.3f} > '
                    f'{self.carried_navigation_radius_limit:.3f} m'
                )
        if not self._fresh_retention_probe(
            'compact_transport',
            'compact_transport_final',
            leg=len(legs),
        ):
            raise RuntimeError('compact_transport_failed')
        self._publish_status(
            'transport_compact',
            command='compact_transport',
            waypoints=len(legs),
            planar_radius=compact_radius,
        )
        if staging_terminal is not None:
            self._carried_staging_solution = staging_terminal
        self._cached_post_retreat_plan = None
        self._post_retreat_shelf_front_x = None
        self._supported_post_retreat_staging_required = False
        return True

    def _place(self, payload=None) -> bool:
        release_only_request = release_only_endpoint = None
        if getattr(self, 'release_only_place_planning_enabled', False):
            if not getattr(self, 'book_centered_place_enabled', False):
                raise RuntimeError('release-only planning requires normal centered PLACE')
        if (getattr(self, 'place_finish_at_release_enabled', False)
                or (payload or {}).get('completion_mode') is not None):
            release_pose_finish.validate_mode(self, payload)
        correlation = None
        direct_empty_home = None
        if getattr(self, 'delivery_evidence_enabled', False):
            identity = AttemptIdentity(**{key: (payload or {}).get(key) for key in
                ('trial_id', 'placement_attempt_id', 'target_model')})
            if identity.target_model != self._target_book_model:
                raise RuntimeError('placement_target_identity_changed')
            correlation = vars(identity)
        if self._held_book_corners is None:
            raise RuntimeError('place requested without a retained book envelope')
        if self._carried_staging_solution is None:
            raise RuntimeError('place requested without a carried staging pose')
        if not self._fresh_retention_probe('place', 'post_navigation'):
            raise RuntimeError('target-book contact was not retained before placement')
        carried_start = self._measured_left_solution()
        bin_surface = self._wait_for_perception_point('latest_bin')
        if not (
            0.30 < bin_surface[0] < 1.35
            and abs(bin_surface[1]) < 0.75
            and 0.70 < bin_surface[2] < 1.75
        ):
            raise RuntimeError(f'bin point outside safe workspace: {np.round(bin_surface, 3)}')
        from .place_input_handoff import await_idle_after_capture
        await_idle_after_capture(self, payload)
        # Arm/table/bin contact evidence starts with this PLACE planning attempt
        # and survives intended release until _run_command's finally cleanup.
        _activate_place_contacts(self, correlation)
        centered_target = None
        if getattr(self, 'book_centered_place_enabled', False):
            centered_target = book_centered_place_target(
                self._held_book_corners, bin_surface,
                self.book_centered_place_height_above_point,
                bin_rotation=(self._selected_place_bin_scene['rotation']
                              if getattr(self, 'bin_scene_required', False) else None),
            )
            release = centered_target.tool_position.copy()
            approach_x = self.book_centered_place_approach_x
        else:
            release = bin_surface.copy()
            release[0] += 0.17
            release[2] += self.place_clearance
            approach_x = self.cartesian_clearance
        above = release.copy()
        above[2] += (max(0.13, getattr(self, 'book_centered_place_clearance_height', 0.40)
                            -self.book_centered_place_height_above_point)
                     if centered_target is not None else 0.13)
        clearance = above.copy()
        clearance[0] = approach_x
        positions = [clearance]
        positions.extend(
            self._interpolate_positions(clearance, above, self.cartesian_step)
        )
        positions.extend(
            self._interpolate_positions(above, release, self.cartesian_step)
        )
        place_torso_target = self.place_torso_height
        measured_torso_target = None
        # The nominal route retains its complete carry-to-torso scene check.
        # The micrometre measured-height shortcut is an explicit experiment.
        if (centered_target is not None and getattr(self, 'table_scene_required', False)
                and getattr(self, 'measured_place_height_enabled', False)):
            measured_torso_target = choose_measured_place_height(
                self, carried_start, self.place_torso_height)
            if measured_torso_target is not None:
                place_torso_target = measured_torso_target
        torso_ready = carried_start.copy()
        torso_ready[0] = place_torso_target
        place_staging = self._carried_staging_solution.copy()
        place_staging[0] = place_torso_target
        gravity_supported = bool(
            getattr(self, '_gravity_supported_payload', False)
        )
        place_scene_diagnostics = None
        if centered_target is not None:
            if getattr(self, 'table_scene_required', False):
                self._active_place_scene_reference = self._selected_place_scene_reference
            if getattr(self, 'release_only_place_planning_enabled', False):
                release_only_request = release_only_place_planning.capture(self, payload)
            # Stored staging selects an IK branch only. The checked physical
            # route starts at measured carry, without the old reverse compact.
            try:
                scene_plan = plan_node_scene_checked_place(
                    self, plan_scene_checked_place, positions, centered_target.tool_rotation,
                    carried_start, torso_ready, place_staging, bin_surface,
                    get_package_share_directory,
                    **({'release_only_request': release_only_request}
                       if release_only_request is not None else {}),
                    table_scene=(getattr(self, '_selected_place_table_scene', None)
                                 if getattr(self, 'table_scene_required', False) else None),
                    **(dict(bin_scene=self._selected_place_bin_scene)
                       if getattr(self, 'bin_scene_required', False) else {}),
                )
            except Exception as exc:
                if getattr(self, 'table_scene_required', False):
                    try:
                        from copy import deepcopy
                        self._publish_status('place_planning_failed', command='place',
                            reason=str(exc), exception_type=type(exc).__name__,
                            planning_diagnostics=deepcopy(getattr(exc, 'diagnostics', None)))
                    except Exception:
                        pass  # Preserve the original planning exception even if telemetry fails.
                raise
            solutions = scene_plan.solutions
            orientation_index = scene_plan.orientation_index
            path_score = scene_plan.path_score
            carried_transition_route = scene_plan.setup
            carried_transition_waypoints = carried_transition_route[:-1]
            unloaded_home_waypoints = scene_plan.unloaded_home
            direct_empty_home = getattr(scene_plan, 'empty_return_from_clearance', None)
            place_scene_diagnostics = scene_plan.diagnostics
            selected_target = getattr(scene_plan, 'selected_target', None)
            if selected_target is not None:
                centered_target = selected_target
                release = selected_target.tool_position.copy()
            if release_only_request is not None:
                release_only_endpoint = release_only_place_planning.require_plan(
                    release_only_request, scene_plan, self, correlation)
            elif getattr(scene_plan, 'release_only_endpoint', None) is not None:
                raise RuntimeError('unexpected release-only plan in ordinary PLACE')
        else:
            if gravity_supported:
                # Recompute the audited support-compensated compact path and use
                # it in reverse.  A generic direct interpolation can rotate the
                # jaw horizontal for many seconds before the bin approach.
                forward_supported = self._supported_compact_goals(place_staging)
                carried_staging_waypoints = [
                    *(waypoint.copy() for waypoint in reversed(forward_supported[:-1])),
                    place_staging,
                ]
            else:
                carried_staging_waypoints = (
                    []
                    if np.allclose(torso_ready[1:], place_staging[1:])
                    else [place_staging]
                )
            if not self._carried_robot_transition_is_safe(
                carried_start,
                torso_ready,
                self._held_book_corners,
            ):
                raise RuntimeError('Carried book would collide while raising the torso')
            if gravity_supported and not self._gravity_supported_transition_is_safe(
                carried_start,
                torso_ready,
            ):
                raise RuntimeError('Carried book loses gravity support while raising the torso')
            carried_staging_route = self._plan_carried_joint_route(
                torso_ready,
                carried_staging_waypoints,
                self._held_book_corners,
                require_gravity_support=gravity_supported,
            )
            if carried_staging_route is None:
                raise RuntimeError('Carried book would collide returning to staging')
            candidate_start = (
                carried_staging_route[-1]
                if carried_staging_route
                else torso_ready
            )
            validated_candidate_route: Optional[List[np.ndarray]] = None

            def carried_candidate_is_safe(
                candidate_solutions: Sequence[np.ndarray],
                candidate_transition: Sequence[np.ndarray],
            ) -> bool:
                nonlocal validated_candidate_route
                route = self._plan_carried_joint_route(
                    candidate_start,
                    (*candidate_transition, candidate_solutions[0]),
                    self._held_book_corners,
                    require_gravity_support=gravity_supported,
                )
                if route is None:
                    return False
                previous = route[-1]
                for candidate_solution in candidate_solutions[1:]:
                    if not self._carried_robot_transition_is_safe(
                        previous,
                        candidate_solution,
                        self._held_book_corners,
                    ):
                        return False
                    if gravity_supported and not (
                        self._gravity_supported_transition_is_safe(
                            previous,
                            candidate_solution,
                        )
                    ):
                        return False
                    previous = candidate_solution
                validated_candidate_route = route
                return True

            solutions, orientation_index, path_score, transition_waypoints = (
                self._solve_cartesian_path(
                    positions,
                    (
                        (centered_target.tool_rotation,)
                        if centered_target is not None
                        else (
                            supported_bin_place_orientations()
                            if gravity_supported
                            else bin_place_orientations()
                        )
                    ),
                    self.place_torso_height,
                    transition_start=place_staging,
                    candidate_validator=carried_candidate_is_safe,
                    first_valid=True,
                    transition_edge_validator=(
                        self._gravity_supported_transition_is_safe
                        if centered_target is not None and gravity_supported else None
                    ),
                    joint_limit_margin=(
                        max(self.place_joint_limit_margin, 0.10)
                        if gravity_supported
                        else self.place_joint_limit_margin
                    ),
                )
            )
            # The retracted planner may change a whole joint at each staging state.
            # Split those controller moves so the carried book rotates gradually
            # and retention is checked after every bounded leg.  Subdivision is
            # refined against the actual grasp-link attitude change.
            if validated_candidate_route is None:
                raise RuntimeError('Carried book would collide during bin transition')
            carried_transition_route = [
                *carried_staging_route,
                *validated_candidate_route,
            ]
            previous = carried_transition_route[-1]
            for solution in solutions[1:]:
                if not self._carried_robot_transition_is_safe(
                    previous,
                    solution,
                    self._held_book_corners,
                ):
                    raise RuntimeError('Carried book would collide during bin approach')
                if gravity_supported and not (
                    self._gravity_supported_transition_is_safe(previous, solution)
                ):
                    raise RuntimeError('Carried book loses gravity support during bin approach')
                previous = solution
            carried_transition_waypoints = carried_transition_route[:-1]
            home_at_place_height = HOME.copy()
            home_at_place_height[0] = self.place_torso_height
            unloaded_home_waypoints = self._plan_retracted_transition(
                torso_ready,
                home_at_place_height,
            )
            if unloaded_home_waypoints is None:
                raise RuntimeError('No safe unloaded route from carry pose to HOME')
        self._publish_status(
            'ik_ready',
            command='place',
            bin_surface=[float(value) for value in bin_surface],
            placement_target_mode=('nominal_book_centered' if centered_target is not None else 'legacy_tool_offset'),
            place_scene=place_scene_diagnostics,
            nominal_book_center=(centered_target.book_center.tolist() if centered_target is not None else None),
            tool_rotation=(centered_target.tool_rotation.tolist() if centered_target is not None else None),
            nominal_attachment_center=(centered_target.attachment_center.tolist() if centered_target is not None else None),
            release=[float(value) for value in release],
            waypoints=len(solutions),
            orientation_index=orientation_index,
            path_score=path_score,
            staged_transition=bool(carried_transition_waypoints),
            transition_waypoints=len(carried_transition_waypoints),
        )
        if release_only_endpoint is not None:
            release_only_endpoint.require(self, correlation, solutions[-1])
        if not self._fresh_retention_probe('place', 'pre_place_motion'):
            raise RuntimeError('target-book contact was not retained before placement')
        if measured_torso_target is not None:
            require_planned_place_height(self, measured_torso_target)
        if not self._best_effort(
            lambda: self._move_torso(place_torso_target, 2.2, allow_completed_hold=True,
                **({'planned_start': measured_torso_target}
                   if measured_torso_target is not None else {}))
        ):
            return self._recover_closed_place(
                cause='torso_motion_failed',
                remaining_approach_legs=(),
                cartesian_solutions=solutions,
                carried_transition_waypoints=carried_transition_waypoints,
                carried_start=torso_ready,
                unloaded_home_waypoints=unloaded_home_waypoints,
                at_carried_start=True,
                ensure_place_torso=True,
                release_allowed=False,
            )
        if not self._fresh_retention_probe('place', 'torso', leg=0):
            return self._recover_closed_place(
                cause='contact_lost_after_torso',
                remaining_approach_legs=(),
                cartesian_solutions=solutions,
                carried_transition_waypoints=carried_transition_waypoints,
                carried_start=torso_ready,
                unloaded_home_waypoints=unloaded_home_waypoints,
                at_carried_start=True,
            )
        # Timing options apply only to this normal accepted, registered PLACE path.
        arm_timing_options = {}
        arm_speed_scale = getattr(self, 'placement_transport_speed_scale', 1.0)
        if ((direct_empty_home is not None or release_only_endpoint is not None) and arm_speed_scale != 1.0
                and getattr(self, 'table_scene_required', False)
                and getattr(self, 'bin_scene_required', False)):
            arm_timing_options['arm_speed_scale'] = checked_arm_speed_scale(arm_speed_scale)
        approach_legs: List[Tuple[Sequence[float], float, str]] = [
            (waypoint, 0.8, 'bin_transition')
            for waypoint in carried_transition_waypoints
        ]
        approach_legs.append((solutions[0], 2.8, 'bin_clearance'))
        approach_legs.extend(
            (solution, 0.65, 'bin_approach')
            for solution in solutions[1:]
        )
        loaded_arm_timing_options = dict(arm_timing_options)
        if 'arm_speed_scale' in loaded_arm_timing_options:
            loaded_arm_timing_options['arm_speed_scale'] = min(
                loaded_arm_timing_options['arm_speed_scale'],
                checked_arm_speed_scale(getattr(self, 'loaded_place_speed_scale_cap', 3.0)))
        if getattr(self, 'bin_clearance_timing_enabled', False):
            loaded_arm_timing_options.update(bin_clearance_normal_options(
                self, correlation, direct_empty_home, approach_legs))
        if release_only_endpoint is not None:
            release_only_endpoint.require(self, correlation, solutions[-1])
        if getattr(self, 'place_transition_stop_enabled', False):
            loaded_arm_timing_options.update(_place_transition_stop.normal_options(
                self, correlation, approach_legs, loaded_arm_timing_options.get('arm_speed_scale', 1.0),
                (place_scene_diagnostics or {}).get('measured_master')))
        if getattr(self, 'release_only_clearance_timing_enabled', False):
            loaded_arm_timing_options.update(_release_clearance.normal_options(
                self, correlation, release_only_endpoint, approach_legs,
                loaded_arm_timing_options.get('transition_stop'),
                loaded_arm_timing_options.get('arm_speed_scale', 1.0)))
        approach_ok, next_leg, contact_lost = self._execute_retained_arm_legs(
            approach_legs,
            'place',
            **loaded_arm_timing_options,
        )
        if not approach_ok:
            return self._recover_closed_place(
                cause='contact_lost' if contact_lost else 'motion_failed',
                remaining_approach_legs=approach_legs[next_leg:],
                cartesian_solutions=solutions,
                carried_transition_waypoints=carried_transition_waypoints,
                carried_start=torso_ready,
                unloaded_home_waypoints=unloaded_home_waypoints,
                release_allowed=False,
            )
        if getattr(self, 'preopen_stationary_enabled', False):
            # Optional admission only; the original checked route/open stays intact.
            from .preopen_stationary import require_stationary_closed_pose
            require_stationary_closed_pose(
                self, correlation, solutions[-1],
                (place_scene_diagnostics or {}).get('measured_master'))
        if release_only_endpoint is not None:
            release_only_endpoint.require(self, correlation, solutions[-1])
        release_owner = (release_pose_finish.prepare(
            self, correlation, solutions[-1], direct_empty_home,
            **({'release_only_endpoint': release_only_endpoint}
               if release_only_endpoint is not None else {}))
            if getattr(self, 'place_finish_at_release_enabled', False) else None)
        if release_only_endpoint is not None and release_owner is None:
            raise RuntimeError('release-only plan cannot enter legacy release/return')
        if correlation is not None:
            # The original open command must attain its measured endpoint
            # before software payload state is cleared or unloaded return starts.
            def verify_open():
                measured = observe_measured_open_pose(self, correlation, solutions[-1],
                    'placement_open_measured')
                if release_owner is not None:
                    return release_pose_finish.remember_open(self, release_owner, measured)
                return measured['verified']
            if not self._open_gripper(verify_measurement=verify_open):
                # Uncertain release: no second opening or unloaded recovery.
                raise RuntimeError('placement_release_unverified')
            if release_owner is not None:
                return release_pose_finish.finish(self, release_owner)
            finish_options, measured_return_goal = raised_place_normal_finish(
                self, correlation, torso_ready, direct_empty_home, HOME)
            if not self._return_from_bin(solutions, carried_transition_waypoints,
                                         torso_ready, unloaded_home_waypoints,
                    **({'direct_empty_home': direct_empty_home}
                       if direct_empty_home is not None else {}), **arm_timing_options, **finish_options):
                return False
            return observe_measured_open_pose(self, correlation, measured_return_goal,
                'placement_hand_return_measured')['verified']
        if not self._open_gripper():
            if getattr(self, 'table_scene_required', False):
                raise RuntimeError('placement_release_unverified')
            return self._recover_closed_place(
                cause='release_failed',
                remaining_approach_legs=(),
                cartesian_solutions=solutions,
                carried_transition_waypoints=carried_transition_waypoints,
                carried_start=torso_ready,
                unloaded_home_waypoints=unloaded_home_waypoints,
            )
        return self._return_from_bin(
            solutions,
            carried_transition_waypoints,
            torso_ready,
            unloaded_home_waypoints,
            **({'direct_empty_home': direct_empty_home}
               if direct_empty_home is not None else {}),
            **arm_timing_options,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ManipulationNode()
    try:
        _mark_planning_cpu_executor(node)
    except Exception:
        pass
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        node._cancel.set()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
