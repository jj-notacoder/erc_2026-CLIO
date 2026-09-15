"""Bounded-recovery mission state machine and competition scoring interface."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple
import uuid

import numpy as np
import rclpy
from geometry_msgs.msg import Point32, PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import ChannelFloat32, PointCloud
from std_msgs.msg import Int32, String
from tf2_ros import Buffer, TransformException, TransformListener

from . import empty_head_timing as _empty_head_timing
from . import mission_raw_contacts as _mission_raw_contacts
from .bin_geometry import bin_message_is_current, bin_message_is_verified
from .release_evidence import AttemptIdentity, ReleaseEvidence
from .release_pose_evidence import (ReleasePoseEvidence, MODE as RELEASE_POSE_MODE,
    EVENT as RELEASE_POSE_EVENT, CONTRACT as RELEASE_EVIDENCE_CONTRACT,
    checked_enabled as checked_release_pose_enabled)
from .shelf_bay_context import make_bay_context
from .bin_visibility_recovery import (
    BinApproachReference,
    RECOVERY_TIMEOUT_NS,
    verify_reacquired_bin,
    visibility_retreat_goal,
)

from .common import (
    SENSOR_QOS,
    RELIABLE_QOS,
    TRANSIENT_RELIABLE_QOS,
    decode_event,
    encode_event,
    quaternion_from_yaw,
    yaw_from_quaternion,
)
from .runtime_utils import (
    ContactEpisodeTracker,
    book_model_name,
    is_intentional_gripper_book_contact,
    is_target_book_bin_contact,
    normalize_contact_pair,
    polyline_length,
    scored_contact_object,
    stamp_to_nanoseconds,
)

import tf2_geometry_msgs  # noqa: F401, E402


PHYSICAL_SHELF_COLUMN_COUNT = 5
PHYSICAL_SHELF_COLUMN_SPACING = 1.0
ACTIVE_BOOK_MODEL_ROW_OFFSET = 1
PHYSICAL_COLUMN_RESIDUAL_LIMIT = 0.35


class MissionManager(Node):
    """Coordinate perception, navigation, and one-arm manipulation end to end."""

    def __init__(self) -> None:
        super().__init__('erc_mission_manager')
        self._declare_parameters()
        self.target_column = int(self.get_parameter('shelf_column_number').value)
        self.target_colour = str(self.get_parameter('book_colour').value).lower()
        self.team_name = str(self.get_parameter('team_name').value)
        self.dry_run = bool(self.get_parameter('dry_run').value)
        self.delivery_evidence_enabled = bool(
            self.get_parameter('delivery_evidence_enabled').value)
        from .head_return_overlap import checked_enabled
        self.head_return_overlap_enabled = checked_enabled(self.get_parameter('head_return_overlap_enabled').value)
        self.empty_head_timing_enabled = _empty_head_timing.checked_enabled(
            self.get_parameter('empty_head_timing_enabled').value)
        self._empty_head_request = None
        self._head_return_request = None
        self.place_finish_at_release_enabled = checked_release_pose_enabled(
            self.get_parameter('place_finish_at_release_enabled').value)
        self.mission_raw_contacts_enabled = _mission_raw_contacts.checked_enabled(
            self.get_parameter('mission_raw_contacts_enabled').value)
        self._mission_raw_contacts_failure = None
        self.delivery_release_evidence = None
        self._delivery_release_snapshot = None
        self.sensor_timeout = float(
            self.get_parameter('sensor_ready_timeout_seconds').value
        )
        self.marker_search_negative_enabled = bool(
            self.get_parameter('marker_search_negative_enabled').value)
        self.marker_search_timeout = float(
            self.get_parameter('marker_search_timeout_seconds').value
        )
        self.perception_timeout = float(
            self.get_parameter('perception_timeout_seconds').value
        )
        self.navigation_timeout = float(
            self.get_parameter('navigation_timeout_seconds').value
        )
        self.manipulation_timeout = float(
            self.get_parameter('manipulation_timeout_seconds').value
        )
        self.shelf_standoff = float(
            self.get_parameter('shelf_standoff_distance').value
        )
        self.grasp_standoff = float(
            self.get_parameter('grasp_standoff_distance').value
        )
        self.top_row_grasp_standoff = float(
            self.get_parameter('top_row_grasp_standoff_distance').value
        )
        if not math.isfinite(self.top_row_grasp_standoff) or not (
            0.45 <= self.top_row_grasp_standoff <= 0.90
        ):
            raise ValueError('top_row_grasp_standoff_distance must be 0.45–0.90 m')
        self.grasp_lateral_bias = float(
            self.get_parameter('grasp_lateral_bias_distance').value
        )
        self.empty_arm_staging_enabled = bool(
            self.get_parameter('empty_arm_staging_enabled').value
        )
        self.empty_arm_staging_backoff = float(
            self.get_parameter('empty_arm_staging_backoff_distance').value
        )
        if not .06 <= self.empty_arm_staging_backoff <= .30:
            raise ValueError('empty_arm_staging_backoff_distance must be 0.06–0.30 m')
        self._empty_stage_pending = False
        self._empty_stage_plan_id = None
        self.lift_first_extraction_enabled = bool(
            self.get_parameter('lift_first_extraction_enabled').value
        )
        self._shelf_registration_stamp_ns = None
        self.bin_standoff = float(
            self.get_parameter('bin_standoff_distance').value
        )
        self.bin_visibility_recovery_enabled = bool(
            self.get_parameter('bin_visibility_recovery_enabled').value
        )
        self._bin_approach_reference = None
        self._bin_visibility_attempts = 0
        self._bin_visibility_started_ns = None
        self._bin_reacquire_after_ns = -1
        self.carried_shelf_retreat = float(
            self.get_parameter('carried_shelf_retreat_distance').value
        )
        if (
            not math.isfinite(self.carried_shelf_retreat)
            or self.carried_shelf_retreat <= 0.0
        ):
            raise ValueError('carried_shelf_retreat_distance must be positive')
        self.max_pick_attempts = int(self.get_parameter('maximum_pick_attempts').value)
        self.max_search_turns = int(self.get_parameter('maximum_search_turns').value)
        self.search_turn = float(self.get_parameter('search_turn_radians').value)
        self.marker_search_first_probe_enabled = self.get_parameter(
            'marker_search_first_probe_enabled').value
        if type(self.marker_search_first_probe_enabled) is not bool:
            raise ValueError('marker_search_first_probe_enabled must be Boolean')
        if self.marker_search_first_probe_enabled and self.search_turn not in (-math.pi / 4.0, -0.7853981634):
            raise ValueError('first-probe search requires the original -pi/4 fallback')
        self.trial_timeout = float(self.get_parameter('trial_timeout_seconds').value)

        self.trial_id = uuid.uuid4().hex[:12]
        self.results_dir = Path(str(self.get_parameter('results_output_dir').value))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.path_output_dir = self.results_dir / 'paths'
        self.path_output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.results_dir / f'trial_{self.trial_id}.jsonl'
        self.summary_path = self.results_dir / f'trial_{self.trial_id}_summary.json'
        self.wall_started = time.monotonic()
        self._initialize_trial_timing()
        self.state = 'WAIT_READY'
        # Nodes may start before their first /clock sample.  Defer the state
        # timer so a non-zero simulator clock cannot look like a readiness
        # timeout on the first tick.
        self.state_started = None
        self.ready = {'perception': False, 'navigation': False, 'manipulation': False}
        self.start_pose: Optional[Tuple[float, float, float]] = None
        self.robot_pose: Optional[Tuple[float, float, float]] = None
        self.marker_cloud: Optional[PointCloud] = None
        self.book_point: Optional[PointStamped] = None
        self._confirmed_book_point_odom = None
        self._confirmed_book_point_stamp_ns = None
        self._book_mode_epoch_ns = None
        self.bin_point: Optional[PointStamped] = None
        self.bin_invalidated_ns = -1
        self.bin_verified_ns = -1
        self.bin_candidate = None
        self.detected_row: Optional[int] = None
        self.target_physical_column: Optional[int] = None
        self.target_book_model: Optional[str] = None
        self.shelf_normal: Optional[np.ndarray] = None
        self.target_marker_odom: Optional[np.ndarray] = None
        self.nav_event: Optional[Dict] = None
        self.manip_event: Optional[Dict] = None
        self.latest_planned_path: Optional[NavPath] = None
        self.latest_executed_path: Optional[NavPath] = None
        self.active_nav_purpose = ''
        self.active_nav_goal_number = 0
        self.active_nav_dispatched_ns = 0
        self.active_nav_request: Dict = {}
        self.active_nav_terminal: Optional[Dict] = None
        self.active_nav_pending = False
        self.navigation_runs: List[Dict] = []
        self._navigation_run_indices: Dict[int, int] = {}
        self.search_turns = 0
        self._marker_search_probe_origin = None
        self.pick_attempts = 0
        self.column_confirmed = False
        self.row_confirmed = False
        self.bin_contact_confirmed = False
        self.active_contacts = set()
        self.contact_tracker = ContactEpisodeTracker(
            separation_gap=0.25, cooldown=1.0
        )
        self.collision_episodes = 0
        self.nav_goals = 0
        self.finished = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.mode_pub = self.create_publisher(
            String, '/erc/perception/mode', TRANSIENT_RELIABLE_QOS
        )
        self.nav_goal_pub = self.create_publisher(
            PoseStamped, '/erc/navigation/goal', RELIABLE_QOS
        )
        self.nav_command_pub = self.create_publisher(
            String, '/erc/navigation/command', RELIABLE_QOS
        )
        self.manip_command_pub = self.create_publisher(
            String, '/erc/manipulation/command', RELIABLE_QOS
        )
        self.status_pub = self.create_publisher(
            String, '/erc/mission/status', TRANSIENT_RELIABLE_QOS
        )
        self.column_pub = self.create_publisher(
            Int32, '/erc/shelf_column_identification', TRANSIENT_RELIABLE_QOS
        )
        self.row_pub = self.create_publisher(
            Int32, '/erc/shelf_row_identification', TRANSIENT_RELIABLE_QOS
        )

        self.create_subscription(Odometry, '/odom', self._on_odom, SENSOR_QOS)
        self.create_subscription(
            PointCloud,
            '/erc/perception/markers',
            self._on_markers,
            TRANSIENT_RELIABLE_QOS,
        )
        self.create_subscription(
            PointStamped,
            '/erc/perception/target_book',
            self._on_book,
            TRANSIENT_RELIABLE_QOS,
        )
        self.create_subscription(
            PointStamped,
            '/erc/perception/bin',
            self._on_bin,
            TRANSIENT_RELIABLE_QOS,
        )
        self.create_subscription(
            Int32,
            '/erc/perception/row',
            self._on_row,
            TRANSIENT_RELIABLE_QOS,
        )
        self.create_subscription(
            String,
            '/erc/perception/status',
            self._on_perception_status,
            TRANSIENT_RELIABLE_QOS,
        )
        self.create_subscription(
            String,
            '/erc/navigation/status',
            self._on_navigation_status,
            TRANSIENT_RELIABLE_QOS,
        )
        self.create_subscription(
            NavPath,
            '/erc/navigation/planned_path',
            self._on_planned_path,
            TRANSIENT_RELIABLE_QOS,
        )
        self.create_subscription(
            NavPath,
            '/erc/navigation/executed_path',
            self._on_executed_path,
            TRANSIENT_RELIABLE_QOS,
        )
        self.create_subscription(
            String,
            '/erc/manipulation/status',
            self._on_manipulation_status,
            TRANSIENT_RELIABLE_QOS,
        )
        if self.mission_raw_contacts_enabled:
            self.create_subscription(Contacts, '/contacts',
                _mission_raw_contacts.mission_raw_contacts_callback(
                    self, self._on_contacts, '/contacts'), SENSOR_QOS, raw=True)
            self.create_subscription(Contacts, '/bin_contacts',
                _mission_raw_contacts.mission_raw_contacts_callback(
                    self, self._on_bin_contacts, '/bin_contacts'), SENSOR_QOS, raw=True)
        else:
            self.create_subscription(Contacts, '/contacts', self._on_contacts, SENSOR_QOS)
            self.create_subscription(
                Contacts, '/bin_contacts', self._on_bin_contacts, SENSOR_QOS
            )
        self.create_timer(0.10, self._tick)
        self.create_timer(0.50, self._republish_score)
        self._log(
            'trial_started',
            team_name=self.team_name,
            shelf_column_number=self.target_column,
            book_colour=self.target_colour,
            dry_run=self.dry_run,
            timing=self._trial_timing(time.monotonic()),
        )

    def _declare_parameters(self) -> None:
        values = {
            'shelf_column_number': 1,
            'book_colour': 'red',
            'team_name': 'TEAM_NAME',
            'dry_run': False,
            'delivery_evidence_enabled': False,
            'place_finish_at_release_enabled': False,
            'mission_raw_contacts_enabled': False,
            'head_return_overlap_enabled': False,
            'empty_head_timing_enabled': False,
            'launch_origin_monotonic_seconds': -1.0,
            'launch_origin_utc': '',
            'launch_origin_basis': 'not_provided',
            'results_output_dir': 'results',
            'sensor_ready_timeout_seconds': 35.0,
            'marker_search_timeout_seconds': 5.0,
            'marker_search_negative_enabled': False,
            'perception_timeout_seconds': 18.0,
            'navigation_timeout_seconds': 50.0,
            'manipulation_timeout_seconds': 180.0,
            'shelf_standoff_distance': 1.30,
            'grasp_standoff_distance': 0.65,
            'top_row_grasp_standoff_distance': 0.65,
            'empty_arm_staging_enabled': False,
            'lift_first_extraction_enabled': False,
            'empty_arm_staging_backoff_distance': 0.20,
            'grasp_lateral_bias_distance': 0.05,
            'bin_standoff_distance': 0.72,
            'bin_visibility_recovery_enabled': False,
            'carried_shelf_retreat_distance': 0.35,
            'maximum_pick_attempts': 1,
            'maximum_search_turns': 8,
            'marker_search_first_probe_enabled': False,
            'search_turn_radians': -math.pi / 4.0,
            'trial_timeout_seconds': 270.0,
        }
        for name, default in values.items():
            self.declare_parameter(name, default)

    def _initialize_trial_timing(self) -> None:
        """Keep an explicit launch origin separate from the legacy mission timer."""
        self.mission_initialization_utc = datetime.now(timezone.utc).isoformat(
            timespec='microseconds'
        )
        self.first_target_bin_contact = None
        raw_seconds = self.get_parameter('launch_origin_monotonic_seconds').value
        utc = str(self.get_parameter('launch_origin_utc').value)
        basis = str(self.get_parameter('launch_origin_basis').value)
        try:
            seconds = float(raw_seconds)
        except (TypeError, ValueError):
            seconds = math.nan
        reason = None
        if not math.isfinite(seconds) or seconds < 0.0:
            reason = 'missing_or_invalid_monotonic_origin'
        elif seconds > self.wall_started:
            reason = 'launch_origin_after_mission_initialization'
        elif not basis.strip() or basis == 'not_provided':
            reason = 'missing_origin_basis'
        else:
            try:
                utc_time = datetime.fromisoformat(utc.replace('Z', '+00:00'))
                offset = utc_time.utcoffset()
                if offset is None or offset.total_seconds() != 0.0:
                    raise ValueError('origin is not timezone-aware UTC')
            except ValueError:
                reason = 'missing_or_invalid_origin_utc'
        self.launch_origin = {
            'status': 'available' if reason is None else 'unavailable',
            'monotonic_seconds': (
                seconds if math.isfinite(seconds) and seconds >= 0.0 else None
            ),
            'utc': utc or None,
            'basis': basis or None,
            'unavailable_reason': reason,
        }

    def _record_first_target_bin_contact(
        self, message: Contacts, target_pairs, target_model: str
    ) -> None:
        """Freeze the first exact named contact receipt, not containment proof.

        Producer ROS time is retained independently: converting it to this host
        monotonic clock would invent an unmeasured wall/simulation-time offset.
        """
        if getattr(self, 'first_target_bin_contact', None) is not None:
            return
        receipt_seconds = time.monotonic()
        receipt_utc = datetime.now(timezone.utc).isoformat(timespec='microseconds')
        stamp = getattr(getattr(message, 'header', None), 'stamp', None)
        producer_ns = stamp_to_nanoseconds(stamp) if stamp is not None else 0
        self.first_target_bin_contact = {
            'receipt_monotonic_seconds': receipt_seconds,
            'receipt_utc': receipt_utc,
            'producer_ros_time_ns': producer_ns if producer_ns > 0 else None,
            'source_topic': '/bin_contacts',
            'target_book_model': target_model,
            'pairs': sorted(target_pairs),
            'mission_state_at_receipt': self.state,
            'endpoint_basis': 'first_exact_named_target_bin_contact_receipt',
        }

    def _trial_timing(self, summary_monotonic_seconds: float) -> Dict:
        """Add timing metadata without redefining historical elapsed_seconds."""
        origin = getattr(self, 'launch_origin', {
            'status': 'unavailable',
            'monotonic_seconds': None,
            'utc': None,
            'basis': None,
            'unavailable_reason': 'launch_origin_not_initialized',
        })
        contact = getattr(self, 'first_target_bin_contact', None)
        dry_run = bool(getattr(self, 'dry_run', False))
        contact_elapsed = None
        if dry_run:
            status = 'excluded_dry_run'
        elif origin['status'] != 'available':
            status = 'unavailable_launch_origin'
        elif contact is None:
            status = 'pending_contact'
        else:
            interval = (
                contact['receipt_monotonic_seconds'] - origin['monotonic_seconds']
            )
            if math.isfinite(interval) and interval >= 0.0:
                contact_elapsed = interval
                status = 'recorded'
            else:
                status = 'invalid_monotonic_order'
        return {
            'schema_version': 1,
            'mode': 'dry_run' if dry_run else 'physical_run',
            'clock_basis': 'host_monotonic_seconds',
            'launch_origin': origin,
            'mission_initialization': {
                'monotonic_seconds': self.wall_started,
                'utc': getattr(self, 'mission_initialization_utc', None),
                'basis': 'mission_manager.wall_started_assignment',
            },
            'first_target_bin_contact': contact,
            'launch_to_first_target_bin_contact_wall_seconds': contact_elapsed,
            'launch_to_first_target_bin_contact_status': status,
            'launch_to_first_target_bin_contact_basis': (
                'host_monotonic_launch_origin_to_first_exact_named_contact_receipt'
            ),
            'placement_verified_by_contact_timer': False,
            'mission_initialization_to_summary_wall_seconds': (
                summary_monotonic_seconds - self.wall_started
            ),
        }

    def _log(self, event: str, **fields) -> None:
        payload = {
            'event': event,
            'state': self.state,
            'trial_id': self.trial_id,
            'wall_time_utc': datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
            'ros_time_ns': self.get_clock().now().nanoseconds,
        }
        payload.update(fields)
        with self.log_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(payload, sort_keys=True) + '\n')

    def _publish_status(self, event: str, **fields) -> None:
        message = String()
        message.data = encode_event(event, state=self.state, trial_id=self.trial_id, **fields)
        self.status_pub.publish(message)
        self._log(event, **fields)

    def _set_state(self, state: str, **fields) -> None:
        previous = self.state
        now = self.get_clock().now()
        elapsed = (
            0.0
            if self.state_started is None
            else (now - self.state_started).nanoseconds / 1e9
        )
        self.state = state
        self.state_started = now
        self._publish_status(
            'state_transition',
            previous=previous,
            current=state,
            previous_elapsed=elapsed,
            **fields,
        )

    def _elapsed_state(self) -> float:
        now = self.get_clock().now()
        if self.state_started is None:
            self.state_started = now
            return 0.0
        if self.state_started.nanoseconds == 0 and now.nanoseconds > 0:
            self.state_started = now
            return 0.0
        return (now - self.state_started).nanoseconds / 1e9

    def _on_odom(self, message: Odometry) -> None:
        pose = message.pose.pose
        self.robot_pose = (
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quaternion(pose.orientation),
        )
        if self.start_pose is None:
            self.start_pose = self.robot_pose

    def _on_markers(self, message: PointCloud) -> None:
        if not _empty_head_timing.mission_observation_current(self, 'markers', message):
            return
        self.marker_cloud = message
        gate = getattr(self, '_marker_search_negative_gate', None)
        if gate is not None:
            gate.reset()  # Positive clouds retain priority over optional absence.

    def _on_book(self, message: PointStamped) -> None:
        if not _empty_head_timing.mission_observation_current(self, 'books', message):
            return
        epoch = getattr(self, '_book_mode_epoch_ns', None)
        stamp_ns = stamp_to_nanoseconds(message.header.stamp)
        now_ns = int(self.get_clock().now().nanoseconds)
        if (epoch is None or stamp_ns <= epoch
                or not 0 <= now_ns-stamp_ns <= 200_000_000):
            return
        self.book_point = message

    def _on_bin(self, message: PointStamped) -> None:
        if bin_message_is_current(
            message, self.get_clock().now().nanoseconds,
            getattr(self, 'bin_invalidated_ns', -1),
        ):
            self.bin_candidate = message
            self.bin_point = message if bin_message_is_verified(
                message, self.get_clock().now().nanoseconds,
                getattr(self, 'bin_verified_ns', -1),
                getattr(self, 'bin_invalidated_ns', -1),
            ) else None

    def _on_row(self, message: Int32) -> None:
        # The overview frame establishes the competition row (top-to-bottom
        # 1..4).  Close-up frames may contain only part of the shelf, so never
        # let their locally assigned labels overwrite the confirmed row.
        if not self.row_confirmed and 1 <= message.data <= 4:
            self.detected_row = int(message.data)

    def _on_perception_status(self, message: String) -> None:
        payload = decode_event(message.data)
        if payload.get('event') == 'ready':
            self.ready['perception'] = True
        if (payload.get('event') == 'marker_search_frame_completed'
                and getattr(self, 'marker_search_negative_enabled', False)
                and self.state == 'SEARCH_COLUMN'):
            gate = getattr(self, '_marker_search_negative_gate', None)
            if gate is not None:
                gate.observe(payload, int(self.get_clock().now().nanoseconds))
        if payload.get('mode') == 'bin' and 'bin_valid' in payload:
            attribute = 'bin_verified_ns' if payload['bin_valid'] is True else 'bin_invalidated_ns'
            setattr(self, attribute, max(
                getattr(self, attribute, -1), int(payload.get('observation_stamp_ns', -1)),
            ))
            candidate = getattr(self, 'bin_candidate', None)
            self.bin_point = candidate if bin_message_is_verified(
                candidate, self.get_clock().now().nanoseconds,
                getattr(self, 'bin_verified_ns', -1),
                getattr(self, 'bin_invalidated_ns', -1),
            ) else None

    def _on_navigation_status(self, message: String) -> None:
        payload = decode_event(message.data)
        event = payload.get('event')
        if event == 'ready':
            self.ready['navigation'] = True
        elif event == 'blocked':
            self.nav_event = payload
            self._log('navigation_event', payload=payload)
        elif event in ('reached', 'failed', 'rejected', 'cancelled'):
            self.nav_event = payload
            self._log('navigation_event', payload=payload)
            if self.active_nav_pending:
                self.active_nav_terminal = payload
                self._persist_navigation_run(payload)
                self.active_nav_pending = False

    def _on_planned_path(self, message: NavPath) -> None:
        if not self._path_belongs_to_active_goal(message):
            return
        if message.poses and self.active_nav_request:
            endpoint = message.poses[-1].pose.position
            if math.hypot(
                float(endpoint.x) - float(self.active_nav_request['x']),
                float(endpoint.y) - float(self.active_nav_request['y']),
            ) > 0.05:
                return
        self.latest_planned_path = message
        if self.active_nav_terminal is not None:
            self._persist_navigation_run(self.active_nav_terminal)

    def _on_executed_path(self, message: NavPath) -> None:
        if not self._path_belongs_to_active_goal(message):
            return
        self.latest_executed_path = message
        if self.active_nav_terminal is not None:
            self._persist_navigation_run(self.active_nav_terminal)

    def _path_belongs_to_active_goal(self, message: NavPath) -> bool:
        if self.active_nav_goal_number <= 0:
            return False
        stamp_ns = stamp_to_nanoseconds(message.header.stamp)
        return stamp_ns == 0 or stamp_ns >= self.active_nav_dispatched_ns

    def _expected_target_book_model(self) -> Optional[str]:
        """Return the one spawned model belonging to the selected shelf cell."""
        physical_column = getattr(self, 'target_physical_column', None)
        detected_row = getattr(self, 'detected_row', None)
        colour = str(getattr(self, 'target_colour', '')).strip().lower()
        if (
            not isinstance(physical_column, int)
            or not 1 <= physical_column <= PHYSICAL_SHELF_COLUMN_COUNT
            or not isinstance(detected_row, int)
            or not 1 <= detected_row <= 4
            or colour not in ('red', 'green', 'yellow', 'blue')
        ):
            return None
        # simulation.launch.py spawns books only on ACTIVE_ROWS [1, 2, 3, 4]
        # and names them with the one-based full-shelf row (2..5).  Perception
        # deliberately reports those four active rows as logical rows 1..4.
        model_row = detected_row + ACTIVE_BOOK_MODEL_ROW_OFFSET
        return f'book_col_{physical_column}_row_{model_row}_{colour}'

    def _target_identity_confirmed(self) -> bool:
        expected_model = self._expected_target_book_model()
        observed_model = book_model_name(
            str(getattr(self, 'target_book_model', '') or '')
        )
        return expected_model is not None and observed_model == expected_model

    def _on_manipulation_status(self, message: String) -> None:
        payload = decode_event(message.data)
        if _empty_head_timing.mission_status(self, payload):
            return
        if payload.get('command') == 'preposition_bin_head':
            from .head_return_overlap import mission_status as head_return_status
            if head_return_status(self, payload):
                return
        from .place_input_handoff import mission_status
        if mission_status(self, payload):
            return
        event = payload.get('event')
        if getattr(self, 'delivery_evidence_enabled', False) and (
                (payload.get('command') == 'place' and event in
                 ('started', 'succeeded', 'failed', 'rejected', 'cancelled'))
                or str(event).startswith('placement_')):
            evidence = self.delivery_release_evidence
            if evidence is None or not evidence.identity.matches(payload):
                self._log('placement_status_rejected', payload=payload, reason='attempt_mismatch')
                return
            now = int(self.get_clock().now().nanoseconds)
            if event == 'placement_open_measured': evidence.opening(payload, now)
            elif event == 'placement_hand_return_measured': evidence.hand_return(payload, now)
            elif event == RELEASE_POSE_EVENT:
                if isinstance(evidence, ReleasePoseEvidence):
                    evidence.finish_pose(payload, now)
                else:
                    evidence.invalidate('unexpected_release_pose_milestone')
            elif event == 'succeeded': evidence.motion_terminal(payload, now, time.monotonic())
            self._delivery_release_snapshot = evidence.evaluate(now, time.monotonic())
            if str(event).startswith('placement_'):
                self._log('placement_measurement', payload=payload)
                return

        if event == 'target_book_latched':
            observed_model = book_model_name(str(payload.get('model', '')))
            expected_model = self._expected_target_book_model()
            current_model = getattr(self, 'target_book_model', None)
            accepting_identity = getattr(self, 'state', 'PICK') == 'PICK'
            if (
                accepting_identity
                and observed_model is not None
                and observed_model == expected_model
                and current_model in (None, observed_model)
            ):
                self.target_book_model = observed_model
            elif accepting_identity:
                self._log(
                    'target_book_identity_rejected',
                    observed_model=observed_model,
                    expected_model=expected_model,
                    reason=(
                        'expected_model_unavailable'
                        if expected_model is None
                        else 'scoped_model_unavailable'
                        if observed_model is None
                        else 'unexpected_model'
                    ),
                )
        if event == 'empty_arm_hazard':
            self._log('manipulation_event', payload=payload)
            if not self.finished:
                self._abort('empty_arm_hazard:' + str(payload.get('reason', 'unknown')))
            return
        if event == 'payload_hazard':
            self._log('manipulation_event', payload=payload)
            carried_states = {
                'PICK',
                'CLEAR_SHELF_WITH_BOOK',
                'COMPACT_TRANSPORT',
                'RETURN_START',
                'HEAD_BIN',
                'FIND_BIN',
                'ALIGN_BIN',
                'HEAD_REACQUIRE_BIN',
                'REACQUIRE_BIN',
                'BIN_VISIBILITY_RETREAT',
                'PLACE',
            }
            if self.state in carried_states and not self.finished:
                reason = str(payload.get('reason', 'unknown'))
                self._abort(f'payload_hazard:{reason}')
            return
        if event == 'ready':
            self.ready['manipulation'] = True
        elif event in ('succeeded', 'failed', 'rejected', 'cancelled'):
            self.manip_event = payload
            self._log('manipulation_event', payload=payload)
        elif event in (
            'ik_target',
            'pick_approach_planned',
            'lower_shelf_pick_candidate_planned',
            'lower_shelf_pick_candidate_rejected',
            'lower_shelf_planning_stage',
            'empty_pickup_geometry_verified',
            'empty_pickup_endpoint_waiting',
            'empty_pickup_endpoint_verified',
            'empty_pickup_endpoint_rejected',
            'ik_ready',
            'gripper_closed',
            'adaptive_gripper_interrupted',
            'fine_gripper_progress',
            'lift_first_extraction_planned',
            'lift_first_geometry_started',
            'lift_first_geometry_progress',
            'lift_first_measured_geometry_verified',
            'lift_first_preflight_rejected',
            'withdrawal_timing_admission_rejected',
            'place_planning_stage',
            'place_planning_failed',
            'placement_torso_motion_admitted',
            'placement_torso_arrival_started',
            'placement_torso_arrival_verified',
            'placement_torso_failed',
            'preclose_aperture_geometry_started',
            'preclose_aperture_geometry_verified',
            'preclose_aperture_geometry_reused',
            'lift_first_pressure_waiting',
            'lift_first_pressure_admitted',
            'lift_first_pressure_rejected',
            'grasp_verified',
            'retention_verified',
            'target_book_latched',
            'grasp_lost',
            'carried_recovery',
            'motion_exception',
            'trajectory_endpoint_missed',
            'transport_compact',
            'empty_arm_plan_ready',
        ):
            # These milestones are deliberately diagnostic-only: keeping them
            # out of manip_event prevents a transient update from satisfying
            # or failing the command-level state-machine checks.
            self._log('manipulation_event', payload=payload)

    @staticmethod
    def _contact_pairs(message: Contacts) -> set:
        pairs = set()
        for contact in getattr(message, 'contacts', []):
            first = getattr(getattr(contact, 'collision1', None), 'name', '')
            second = getattr(getattr(contact, 'collision2', None), 'name', '')
            if first or second:
                pairs.add(normalize_contact_pair(first, second))
        return pairs

    def _on_contacts(self, message: Contacts) -> None:
        evidence = getattr(self, 'delivery_release_evidence', None)
        if getattr(self, 'delivery_evidence_enabled', False) and isinstance(evidence, ReleasePoseEvidence):
            now = int(self.get_clock().now().nanoseconds)
            evidence.observe_contact_message(message, now, source_topic='/contacts')
            self._delivery_release_snapshot = evidence.evaluate(now, time.monotonic())
        timestamp = self.get_clock().now().nanoseconds / 1e9
        object_keys = set()
        observed_pairs = self._contact_pairs(message)
        contact_target_model = (
            getattr(self, 'target_book_model', None)
            or self._expected_target_book_model()
        )
        for pair in observed_pairs:
            if is_intentional_gripper_book_contact(
                pair,
                self.target_colour,
                contact_target_model,
            ):
                continue
            contacted_object = scored_contact_object(pair)
            if contacted_object is not None:
                object_keys.add(('', contacted_object))
        started = self.contact_tracker.observe(object_keys, timestamp)
        self.active_contacts = self.contact_tracker.active_pairs(timestamp)
        if started:
            self.collision_episodes += len(started)
            self._log(
                'unexpected_contact',
                objects=[pair[1] for pair in sorted(started)],
                source_pairs=sorted(observed_pairs),
                count=self.collision_episodes,
            )

    def _on_bin_contacts(self, message: Contacts) -> None:
        target_model = getattr(self, 'target_book_model', None)
        if not target_model:
            return
        evidence = getattr(self, 'delivery_release_evidence', None)
        if getattr(self, 'delivery_evidence_enabled', False) and isinstance(evidence, ReleasePoseEvidence):
            now = int(self.get_clock().now().nanoseconds)
            evidence.observe_contact_message(message, now, source_topic='/bin_contacts')
            self._delivery_release_snapshot = evidence.evaluate(now, time.monotonic())
        pairs = self._contact_pairs(message)
        target_pairs = {
            pair
            for pair in pairs
            if is_target_book_bin_contact(
                pair,
                self.target_colour,
                target_model,
            )
        }
        evidence = getattr(self, 'delivery_release_evidence', None)
        if getattr(self, 'delivery_evidence_enabled', False) and evidence is not None:
            producer_stamp = int(message.header.stamp.sec)*1_000_000_000+int(message.header.stamp.nanosec)
            now = int(self.get_clock().now().nanoseconds)
            if not isinstance(evidence, ReleasePoseEvidence):
                evidence.contact(producer_stamp, now, exact_target_bin_pair=bool(target_pairs))
            self._delivery_release_snapshot = evidence.evaluate(now, time.monotonic())
        if target_pairs:
            self._record_first_target_bin_contact(message, target_pairs, target_model)
            # Timing observes any exact named contact after identity is latched;
            # the existing mission completion gate remains placement-state only.
            if self.state in ('PLACE', 'WAIT_BIN_CONTACT'):
                self.bin_contact_confirmed = True
                self._log(
                    'bin_contact', pairs=sorted(target_pairs),
                    first_target_bin_contact=self.first_target_bin_contact,
                )

    def _republish_score(self) -> None:
        if self.column_confirmed:
            self.column_pub.publish(Int32(data=self.target_column))
        if self.row_confirmed and self.detected_row is not None:
            self.row_pub.publish(Int32(data=self.detected_row))

    def _command(self, publisher, event: str, **fields) -> None:
        message = String()
        message.data = encode_event(event, trial_id=self.trial_id, **fields)
        publisher.publish(message)
        self._log('command', target=publisher.topic_name, command=event, fields=fields)

    def _marker_search_probe_yaw(self, index: int, measured_yaw: float) -> float:
        # Nominal stop headings are anchored once in this mission's odom frame.
        # Each current x/y is still supplied by ordinary live odometry. Endpoint
        # yaw error must not accumulate into the fallback schedule.
        if (getattr(self, 'marker_search_first_probe_enabled', False) is not True
                or type(index) is not int or not 0 <= index < 8
                or self.search_turn not in (-math.pi / 4.0, -0.7853981634)
                or not math.isfinite(measured_yaw)):
            raise ValueError('invalid first-probe search configuration or index')
        origin = getattr(self, '_marker_search_probe_origin', None)
        if index == 0:
            if origin is not None:
                raise ValueError('first-probe search cannot reset a started schedule')
            origin = (self.trial_id, float(measured_yaw))
            self._marker_search_probe_origin = origin
        elif (type(origin) is not tuple or len(origin) != 2
                or origin[0] != self.trial_id or not math.isfinite(origin[1])):
            raise ValueError('first-probe search origin is missing or from another trial')
        offset = -math.pi / 3.0 if index == 0 else -index * math.pi / 4.0
        yaw = origin[1] + offset
        return math.atan2(math.sin(yaw), math.cos(yaw))

    def _marker_search_negative_ready(self) -> bool:
        gate = getattr(self, '_marker_search_negative_gate', None)
        if (not getattr(self, 'marker_search_negative_enabled', False)
                or self.state != 'SEARCH_COLUMN' or gate is None):
            return False
        request = gate.request
        if (request.trial_id, request.target_column, request.target_colour) != (
                self.trial_id, self.target_column, self.target_colour):
            return False
        return gate.ready(int(self.get_clock().now().nanoseconds))

    def _perception_mode(self, mode: str) -> None:
        fields = {
            'shelf_column_number': self.target_column,
            'book_colour': self.target_colour,
        }
        if mode == 'books' and self.row_confirmed and self.detected_row is not None:
            fields['confirmed_row'] = self.detected_row
        if mode == 'books':
            from .book_selection_context import make_context
            try:
                fields['book_selection_context'] = make_context(
                    self.target_marker_odom, self.shelf_normal,
                    self._shelf_registration_stamp_ns,
                    int(self.get_clock().now().nanoseconds),
                    self.target_column, self.target_colour,
                    confirmed_row=fields.get('confirmed_row'),
                    confirmed_point=(self._confirmed_book_point_odom
                                     if self.row_confirmed else None),
                    confirmed_stamp_ns=(self._confirmed_book_point_stamp_ns
                                        if self.row_confirmed else None))
            except (TypeError, ValueError) as error:
                self._abort(f'book_selection_context_invalid:{error}')
                return False
            self._book_mode_epoch_ns = fields['book_selection_context']['dispatch_stamp_ns']
            self.book_point = None
        if getattr(self, 'marker_search_negative_enabled', False):
            from .marker_search_completion import NegativeFrameGate, request_from_fields
            self._marker_search_negative_gate = None
            if mode == 'markers':
                self._marker_search_epoch = getattr(self, '_marker_search_epoch', 0)+1
                optional = dict(fields, trial_id=self.trial_id,
                    marker_search_request_id=self._marker_search_epoch,
                    marker_search_not_before_ns=int(self.get_clock().now().nanoseconds))
                request = request_from_fields(optional)
                if request is not None:
                    self._marker_search_negative_gate = NegativeFrameGate(request)
                    fields.update({key:value for key,value in request.fields().items() if key != 'trial_id'})
        if getattr(self, '_head_return_request', None) is not None:
            from .head_return_overlap import bin_epoch
            bin_epoch(self, mode, fields)
        _empty_head_timing.mission_mode(self, mode, fields)
        self._command(self.mode_pub, mode, **fields)

    def _manipulate(self, command: str, **fields) -> None:
        _empty_head_timing.mission_request(self, command, fields)
        if command == 'place' and getattr(self, 'delivery_evidence_enabled', False):
            identity = AttemptIdentity(self.trial_id, uuid.uuid4().hex, self.target_book_model)
            if getattr(self, 'place_finish_at_release_enabled', False):
                self.delivery_release_evidence = ReleasePoseEvidence(identity)
                fields['completion_mode'] = RELEASE_POSE_MODE
                fields['release_evidence_contract'] = RELEASE_EVIDENCE_CONTRACT
            else:
                self.delivery_release_evidence = ReleaseEvidence(identity)
            self._delivery_release_snapshot = None
            # _command already supplies trial_id. Never pass it twice.
            fields.update(placement_attempt_id=identity.placement_attempt_id,
                          target_model=identity.target_model)
        if command == 'place':
            from .place_input_handoff import begin_request
            begin_request(self, fields)
        self.manip_event = None
        self._command(self.manip_command_pub, command, **fields)

    def _navigate(
        self,
        x: float,
        y: float,
        yaw: float,
        purpose: str,
        *,
        profile: str = 'normal',
        **profile_fields,
    ) -> None:
        goal = PoseStamped()
        goal.header.frame_id = 'odom'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.orientation = quaternion_from_yaw(yaw)
        self.nav_event = None
        self.nav_goals += 1
        self.latest_planned_path = None
        self.latest_executed_path = None
        self.active_nav_purpose = purpose
        self.active_nav_goal_number = self.nav_goals
        self.active_nav_dispatched_ns = stamp_to_nanoseconds(goal.header.stamp)
        self.active_nav_request = {
            'x': float(x),
            'y': float(y),
            'yaw': float(yaw),
            'profile': profile,
            **profile_fields,
        }
        self.active_nav_terminal = None
        self.active_nav_pending = True
        if profile == 'normal':
            self.nav_goal_pub.publish(goal)
        else:
            # A single reliable command binds the fragile-payload motion profile
            # to this exact goal.  Publishing a separate one-shot arm message
            # could silently fall back to normal motion after a navigator restart
            # or cross-topic delivery race.
            self._command(
                self.nav_command_pub,
                'navigate',
                x=float(x),
                y=float(y),
                yaw=float(yaw),
                profile=profile,
                **profile_fields,
            )
        self._log(
            'navigation_goal',
            purpose=purpose,
            x=x,
            y=y,
            yaw=yaw,
            profile=profile,
        )

    @staticmethod
    def _path_samples(message: Optional[NavPath]) -> List[Dict]:
        if message is None:
            return []
        fallback_stamp_ns = stamp_to_nanoseconds(message.header.stamp)
        samples: List[Dict] = []
        for stamped_pose in message.poses:
            stamp_ns = stamp_to_nanoseconds(stamped_pose.header.stamp)
            if stamp_ns == 0:
                stamp_ns = fallback_stamp_ns
            pose = stamped_pose.pose
            samples.append(
                {
                    'stamp_ns': stamp_ns,
                    'x': float(pose.position.x),
                    'y': float(pose.position.y),
                    'yaw': yaw_from_quaternion(pose.orientation),
                }
            )
        return samples

    def _persist_navigation_run(self, outcome: Dict) -> None:
        goal_number = self.active_nav_goal_number
        if goal_number <= 0:
            return
        planned = self._path_samples(self.latest_planned_path)
        executed = self._path_samples(self.latest_executed_path)
        planned_length = polyline_length(
            [(sample['x'], sample['y']) for sample in planned]
        )
        executed_length = polyline_length(
            [(sample['x'], sample['y']) for sample in executed]
        )
        efficiency = None
        if planned_length > 1e-6 and executed:
            efficiency = executed_length / planned_length

        purpose = self.active_nav_purpose or 'unspecified'
        safe_purpose = ''.join(
            character if character.isalnum() else '_' for character in purpose
        ).strip('_') or 'unspecified'
        artifact = self.path_output_dir / (
            f'trial_{self.trial_id}_navigation_goal_'
            f'{goal_number:02d}_{safe_purpose}.json'
        )
        payload = {
            'schema_version': 1,
            'trial_id': self.trial_id,
            'saved_at_utc': datetime.now(timezone.utc).isoformat(
                timespec='milliseconds'
            ),
            'mission_goal_number': goal_number,
            'navigator_goal_id': outcome.get('goal_id'),
            'purpose': purpose,
            'requested_goal': self.active_nav_request,
            'outcome': outcome.get('event', 'unknown'),
            'reason': outcome.get('reason', ''),
            'elapsed_seconds': outcome.get('elapsed_seconds'),
            'planned_path': {
                'frame_id': getattr(
                    getattr(self.latest_planned_path, 'header', None),
                    'frame_id',
                    '',
                ),
                'sample_count': len(planned),
                'length_m': planned_length,
                'samples': planned,
            },
            'executed_path': {
                'frame_id': getattr(
                    getattr(self.latest_executed_path, 'header', None),
                    'frame_id',
                    '',
                ),
                'sample_count': len(executed),
                'length_m': executed_length,
                'samples': executed,
            },
            'executed_to_planned_length_ratio': efficiency,
        }
        temporary = artifact.with_suffix('.json.tmp')
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
        )
        temporary.replace(artifact)

        record = {
            'mission_goal_number': goal_number,
            'navigator_goal_id': outcome.get('goal_id'),
            'purpose': purpose,
            'outcome': outcome.get('event', 'unknown'),
            'reason': outcome.get('reason', ''),
            'planned_samples': len(planned),
            'executed_samples': len(executed),
            'planned_length_m': planned_length,
            'executed_length_m': executed_length,
            'executed_to_planned_length_ratio': efficiency,
            'artifact': artifact.relative_to(self.results_dir).as_posix(),
        }
        existing_index = self._navigation_run_indices.get(goal_number)
        if existing_index is None:
            self._navigation_run_indices[goal_number] = len(self.navigation_runs)
            self.navigation_runs.append(record)
        else:
            self.navigation_runs[existing_index] = record
        self._log('navigation_path_saved', **record)

    def _point_to_odom(self, header, point: Point32) -> np.ndarray:
        stamped = PointStamped()
        stamped.header = header
        stamped.point.x = float(point.x)
        stamped.point.y = float(point.y)
        stamped.point.z = float(point.z)
        transformed = self.tf_buffer.transform(
            stamped,
            'odom',
            timeout=Duration(seconds=1.0),
        )
        return np.asarray(
            [transformed.point.x, transformed.point.y, transformed.point.z], dtype=float
        )

    @staticmethod
    def _channel(cloud: PointCloud, name: str) -> Optional[ChannelFloat32]:
        return next((channel for channel in cloud.channels if channel.name == name), None)

    @staticmethod
    def _physical_shelf_column(
        target: Sequence[float],
        shelf_normal: Sequence[float],
        start_xy: Sequence[float],
    ) -> int:
        """Map a marker centre to the physical ``book_col_N`` model index.

        The official launcher assigns model column 1 to the leftmost shelf
        column as seen by the robot at its start pose, then increments toward
        the right.  Marker digits are independently shuffled, so the requested
        competition digit cannot itself be used as the model column number.
        """
        target_xy = np.asarray(target, dtype=float)[:2]
        normal = np.asarray(shelf_normal, dtype=float)
        origin = np.asarray(start_xy, dtype=float)
        if (
            target_xy.shape != (2,)
            or normal.shape != (2,)
            or origin.shape != (2,)
            or not np.all(np.isfinite(target_xy))
            or not np.all(np.isfinite(normal))
            or not np.all(np.isfinite(origin))
        ):
            raise ValueError('physical shelf-column geometry is invalid')
        normal_magnitude = float(np.linalg.norm(normal))
        if normal_magnitude < 1e-9:
            raise ValueError('physical shelf-column normal is invalid')
        normal /= normal_magnitude
        leftward_tangent = np.asarray([normal[1], -normal[0]], dtype=float)
        lateral_offset = float(np.dot(target_xy - origin, leftward_tangent))
        centre_index = 0.5 * (PHYSICAL_SHELF_COLUMN_COUNT - 1)
        estimated_zero_based = (
            centre_index
            - lateral_offset / PHYSICAL_SHELF_COLUMN_SPACING
        )
        zero_based = int(round(estimated_zero_based))
        if not 0 <= zero_based < PHYSICAL_SHELF_COLUMN_COUNT:
            raise ValueError('selected marker lies outside the physical shelf columns')
        expected_lateral_offset = (
            centre_index - zero_based
        ) * PHYSICAL_SHELF_COLUMN_SPACING
        if (
            abs(lateral_offset - expected_lateral_offset)
            > PHYSICAL_COLUMN_RESIDUAL_LIMIT
        ):
            raise ValueError(
                'selected marker is ambiguous between physical shelf columns'
            )
        return zero_based + 1

    def _shelf_geometry(
        self,
        cloud: PointCloud,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        digit_channel = self._channel(cloud, 'digit')
        if digit_channel is None or len(digit_channel.values) != len(cloud.points):
            raise ValueError('marker cloud has no digit channel')
        transformed: List[np.ndarray] = []
        digits: List[int] = []
        for point, digit in zip(cloud.points, digit_channel.values):
            transformed.append(self._point_to_odom(cloud.header, point))
            digits.append(int(round(digit)))
        if self.target_column not in digits:
            raise ValueError('requested digit is absent from marker cloud')
        xy = np.asarray([point[:2] for point in transformed])
        if len(xy) < 3:
            raise ValueError('at least three markers are required to fit the shelf')
        centered = xy - xy.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        tangent = vh[0]
        normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
        target = transformed[digits.index(self.target_column)]
        start_xy = np.asarray(self.start_pose[:2] if self.start_pose else self.robot_pose[:2])
        if float(np.dot(start_xy - target[:2], normal)) < 0.0:
            normal *= -1.0
        normal /= max(1e-9, np.linalg.norm(normal))
        physical_column = self._physical_shelf_column(
            target,
            normal,
            start_xy,
        )
        return target, normal, physical_column

    def _goal_from_target(
        self,
        target: np.ndarray,
        normal: np.ndarray,
        standoff: float,
        lateral_bias: float = 0.0,
    ) -> Tuple[float, float, float]:
        # With the robot facing ``-normal``, [normal_y, -normal_x] is its
        # leftward shelf tangent.  Biasing the base in that direction keeps
        # the target slightly to the right (negative local y) for the left-arm
        # top-row IK branch.  Preserve the shelf-normal heading instead of
        # toeing the base back toward the laterally offset target.
        tangent_left = np.asarray([normal[1], -normal[0]], dtype=float)
        goal_xy = target[:2] + normal * standoff + tangent_left * lateral_bias
        yaw = math.atan2(-normal[1], -normal[0])
        return float(goal_xy[0]), float(goal_xy[1]), yaw

    def _goal_from_robot_side(
        self, target: np.ndarray, standoff: float
    ) -> Tuple[float, float, float]:
        """Approach a target from the robot side at a fixed standoff."""
        if self.robot_pose is None:
            raise ValueError('robot pose is unavailable')
        robot_xy = np.asarray(self.robot_pose[:2], dtype=float)
        normal = robot_xy - np.asarray(target[:2], dtype=float)
        distance = float(np.linalg.norm(normal))
        if distance < 1e-6:
            raise ValueError('target and robot positions coincide')
        return self._goal_from_target(target, normal / distance, standoff)

    def _carried_shelf_retreat_goal(self) -> Tuple[float, float, float]:
        """Move straight away from the shelf before turning a protruding book."""
        if self.robot_pose is None or self.shelf_normal is None:
            raise ValueError('robot pose and shelf normal are required')
        normal = np.asarray(self.shelf_normal, dtype=float)
        robot_xy = np.asarray(self.robot_pose[:2], dtype=float)
        robot_yaw = float(self.robot_pose[2])
        magnitude = float(np.linalg.norm(normal))
        if (
            normal.shape != (2,)
            or not np.all(np.isfinite(normal))
            or magnitude < 1e-9
        ):
            raise ValueError('shelf normal is invalid')
        if robot_xy.shape != (2,) or not np.all(np.isfinite(robot_xy)):
            raise ValueError('robot pose is invalid')
        if not math.isfinite(robot_yaw):
            raise ValueError('robot heading is invalid')
        normal /= magnitude
        goal_xy = robot_xy + normal * self.carried_shelf_retreat
        return float(goal_xy[0]), float(goal_xy[1]), robot_yaw

    def _try_bin_visibility_recovery(self) -> bool:
        """Use one bounded outward move, then require fresh bin reacquisition."""
        if not getattr(self, 'bin_visibility_recovery_enabled', False):
            return False
        if getattr(self, '_bin_visibility_attempts', 0) != 0:
            return False
        if not self._target_identity_confirmed() or not self._nav_reached():
            self._log('bin_visibility_recovery_rejected', reason='unconfirmed_payload_or_navigation')
            return False
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom', 'base_footprint', Time(), timeout=Duration(seconds=0.10),
            )
            translation = transform.transform.translation
            quaternion = transform.transform.rotation
            rotation = np.asarray([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
            norm = float(np.linalg.norm(rotation))
            if not np.all(np.isfinite(rotation)) or norm < 1e-8:
                raise ValueError('bin_recovery_robot_orientation_invalid')
            qx, qy, qz, qw = rotation / norm
            yaw = math.atan2(2.0*(qw*qz+qx*qy), 1.0-2.0*(qy*qy+qz*qz))
            now_ns = self.get_clock().now().nanoseconds
            goal = visibility_retreat_goal(
                getattr(self, '_bin_approach_reference', None),
                (translation.x, translation.y, yaw),
                stamp_to_nanoseconds(transform.header.stamp), now_ns,
                getattr(self, '_bin_visibility_attempts', 0),
            )
        except (TransformException, ValueError) as exc:
            self._log('bin_visibility_recovery_rejected', reason=str(exc))
            return False
        self._bin_visibility_attempts = 1
        self._bin_visibility_started_ns = now_ns
        self.bin_point = self.bin_candidate = None
        self._perception_mode('idle')
        self._navigate(*goal, 'bin_visibility_retreat', profile='carried_retreat')
        self._set_state('BIN_VISIBILITY_RETREAT')
        return True

    def _verify_bin_reacquisition(self) -> bool:
        """Match a new verified point to the original observed bin location."""
        now_ns = self.get_clock().now().nanoseconds
        if not bin_message_is_verified(
            self.bin_point, now_ns,
            getattr(self, 'bin_verified_ns', -1),
            getattr(self, 'bin_invalidated_ns', -1),
        ):
            return False
        stamp_ns = stamp_to_nanoseconds(self.bin_point.header.stamp)
        if stamp_ns <= getattr(self, '_bin_reacquire_after_ns', -1):
            return False
        point = self.tf_buffer.transform(
            self.bin_point, 'odom', timeout=Duration(seconds=0.10),
        ).point
        verify_reacquired_bin(
            getattr(self, '_bin_approach_reference', None),
            (point.x, point.y, point.z), stamp_ns, now_ns,
            getattr(self, '_bin_reacquire_after_ns', -1),
        )
        return True

    def _abort(self, reason: str) -> None:
        if self.finished:
            return
        self.finished = True
        if self.active_nav_pending:
            self.active_nav_terminal = {
                'event': 'mission_aborted',
                'goal_id': self.active_nav_goal_number,
                'reason': reason,
            }
            self._persist_navigation_run(self.active_nav_terminal)
            self.active_nav_pending = False
        from .place_input_handoff import finish_request
        finish_request(self)
        self._command(self.nav_command_pub, 'cancel')
        self._command(self.manip_command_pub, 'cancel')
        self._set_state('ABORTED', reason=reason)
        self._write_summary(False, reason)

    def _complete(self) -> None:
        if self.finished:
            return
        completion = {}
        if getattr(self, 'delivery_evidence_enabled', False) and not self.dry_run:
            evidence = self.delivery_release_evidence
            snapshot = None if evidence is None else evidence.evaluate(
                int(self.get_clock().now().nanoseconds), time.monotonic())
            self._delivery_release_snapshot = snapshot
            if not snapshot or not snapshot['operation_completed']:
                return
            completion = {key: snapshot[key] for key in
                ('success_scope', 'delivery_outcome', 'physical_inside_verified')}
        self.finished = True
        self._set_state('DONE', bin_contact=self.bin_contact_confirmed, **completion)
        self._write_summary(True, '')

    def _write_summary(
        self, success: bool, reason: str, *, log_event: bool = True
    ) -> None:
        expected_target_book_model = self._expected_target_book_model()
        summary_monotonic_seconds = time.monotonic()
        summary = {
            'trial_id': self.trial_id,
            'team_name': self.team_name,
            'shelf_column_number': self.target_column,
            'book_colour': self.target_colour,
            'dry_run': bool(getattr(self, 'dry_run', False)),
            'success': success,
            'failure_reason': reason,
            'elapsed_seconds': summary_monotonic_seconds - self.wall_started,
            'elapsed_seconds_basis': 'mission_initialization_to_summary_wall_monotonic',
            'timing': self._trial_timing(summary_monotonic_seconds),
            'detected_row': self.detected_row,
            'column_identified': self.column_confirmed,
            'row_identified': self.row_confirmed,
            'pick_attempts': self.pick_attempts,
            'navigation_goals': self.nav_goals,
            'navigation_runs': self.navigation_runs,
            'collision_episodes': self.collision_episodes,
            'bin_contact_confirmed': self.bin_contact_confirmed,
            'target_book_model': getattr(self, 'target_book_model', None),
            'expected_target_book_model': expected_target_book_model,
            'target_identity_confirmed': self._target_identity_confirmed(),
        }
        if getattr(self, 'delivery_evidence_enabled', False):
            snapshot = getattr(self, '_delivery_release_snapshot', None)
            summary['delivery_release_evidence'] = snapshot
            summary['success_scope'] = ('dry_run_state_machine' if self.dry_run else
                'placement_operation_completed' if success and snapshot and snapshot.get('operation_completed')
                else None)
            summary['delivery_outcome'] = (snapshot or {}).get('delivery_outcome')
            summary['physical_inside_verified'] = None
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8'
        )
        if log_event:
            self._log('trial_finished', **summary)

    def _finalize_without_ros(self, reason: str) -> None:
        """Persist a terminal summary after Humble has invalidated the context."""
        if self.finished:
            return
        self.finished = True
        self.state = 'ABORTED'
        self._place_input_request = None
        # Publishing cancellation/status messages is no longer legal after
        # SIGINT has shut down the rclpy context.  The report artifact is plain
        # file I/O, so preserve it without attempting any ROS operations.
        self._write_summary(False, reason, log_event=False)

    def _tick(self) -> None:
        if self.finished:
            return
        total_elapsed = time.monotonic() - self.wall_started
        if total_elapsed > self.trial_timeout:
            self._abort('trial_timeout')
            return

        recovery_started = getattr(self, '_bin_visibility_started_ns', None)
        if recovery_started is not None and self.state in (
            'BIN_VISIBILITY_RETREAT', 'HEAD_REACQUIRE_BIN', 'REACQUIRE_BIN',
        ):
            recovery_elapsed = self.get_clock().now().nanoseconds - recovery_started
            if not 0 <= recovery_elapsed <= RECOVERY_TIMEOUT_NS:
                self._abort('bin_visibility_recovery_timeout')
                return

        if self.state == 'WAIT_READY':
            if all(self.ready.values()) and self.start_pose is not None:
                self._manipulate('stow')
                self._set_state('INIT_STOW')
            elif self._elapsed_state() > self.sensor_timeout:
                missing = [name for name, ready in self.ready.items() if not ready]
                self._abort('readiness_timeout:' + ','.join(missing))
            return

        if self.state == 'INIT_STOW':
            if self._manip_succeeded('stow'):
                self._manipulate('look_markers')
                self._set_state('HEAD_MARKERS')
            elif self._manip_failed() or self._elapsed_state() > self.manipulation_timeout:
                self._abort('initial_stow_failed')
            return

        if self.state == 'HEAD_MARKERS':
            if self._manip_succeeded('look_markers'):
                self.marker_cloud = None
                self.target_physical_column = None
                self._perception_mode('markers')
                self._set_state('SEARCH_COLUMN')
            elif self._manip_failed() or self._elapsed_state() > self.manipulation_timeout:
                self._abort('head_marker_pose_failed')
            return

        if self.state == 'SEARCH_COLUMN':
            if self.marker_cloud is not None:
                try:
                    target, normal, physical_column = self._shelf_geometry(
                        self.marker_cloud
                    )
                except (ValueError, TransformException) as exc:
                    self._log('marker_geometry_rejected', reason=str(exc))
                    self.marker_cloud = None
                else:
                    self.target_marker_odom = target
                    self.shelf_normal = normal
                    self.target_physical_column = physical_column
                    self._shelf_registration_stamp_ns = stamp_to_nanoseconds(
                        self.marker_cloud.header.stamp
                    )
                    self.column_confirmed = True
                    self._republish_score()
                    x, y, yaw = self._goal_from_target(target, normal, self.shelf_standoff)
                    self._navigate(x, y, yaw, 'shelf_observation_pose')
                    self._set_state('NAVIGATE_SHELF')
                    return
            early_marker_turn = (getattr(self, 'marker_search_negative_enabled', False)
                                 and self._marker_search_negative_ready())
            if self._elapsed_state() > self.marker_search_timeout or early_marker_turn:
                search_limit = (min(self.max_search_turns, 8)
                                if getattr(self, 'marker_search_first_probe_enabled', False)
                                else self.max_search_turns)
                if self.search_turns >= search_limit or self.robot_pose is None:
                    self._abort('shelf_markers_not_found')
                else:
                    if early_marker_turn and self._elapsed_state() <= self.marker_search_timeout:
                        gate = self._marker_search_negative_gate
                        self._log('marker_search_early_turn',
                            marker_search_request_id=gate.request.request_id,
                            negative_frames=gate.count, last_frame_sequence=gate.last_sequence,
                            rgb_stamp_ns=gate.last_rgb_ns, depth_stamp_ns=gate.last_depth_ns,
                            completed_ros_ns=gate.last_completed_ns, wait_seconds=self._elapsed_state())
                    x, y, yaw = self.robot_pose
                    target_yaw = yaw + self.search_turn
                    if getattr(self, 'marker_search_first_probe_enabled', False):
                        try:
                            target_yaw = self._marker_search_probe_yaw(self.search_turns, yaw)
                        except (TypeError, ValueError) as exc:
                            self._log('marker_search_schedule_rejected', reason=str(exc))
                            self._abort('visual_search_schedule_invalid')
                            return
                    self.search_turns += 1
                    self._navigate(x, y, target_yaw, 'visual_search_turn')
                    self._set_state('SEARCH_TURN')
            return

        if self.state == 'SEARCH_TURN':
            if self._nav_reached():
                self.marker_cloud = None
                if getattr(self, 'marker_search_negative_enabled', False):
                    self._perception_mode('markers')
                self._set_state('SEARCH_COLUMN')
            elif self._nav_failed() or self._elapsed_state() > self.navigation_timeout:
                self._abort('visual_search_motion_failed')
            return

        if self.state == 'NAVIGATE_SHELF':
            if self._nav_reached():
                self._manipulate('look_books')
                self._set_state('HEAD_BOOKS')
            elif self._nav_failed() or self._elapsed_state() > self.navigation_timeout:
                self._abort('navigation_to_shelf_failed')
            return

        if self.state == 'HEAD_BOOKS':
            if self._manip_succeeded('look_books'):
                self.book_point = None
                self.detected_row = None
                self._confirmed_book_point_odom = None
                self._confirmed_book_point_stamp_ns = None
                if self._perception_mode('books') is False:
                    return
                self._set_state('FIND_BOOK')
            elif self._manip_failed() or self._elapsed_state() > self.manipulation_timeout:
                self._abort('head_book_pose_failed')
            return

        if self.state == 'FIND_BOOK':
            if self.book_point is not None and self.detected_row is not None:
                try:
                    point = self.tf_buffer.transform(
                        self.book_point,
                        'odom',
                        timeout=Duration(seconds=1.0),
                    )
                except TransformException as exc:
                    self._log('book_geometry_rejected', reason=str(exc))
                    self.book_point = None
                else:
                    self.row_confirmed = True
                    self._republish_score()
                    target = np.asarray([point.point.x, point.point.y, point.point.z])
                    self._confirmed_book_point_odom = target.copy()
                    self._confirmed_book_point_stamp_ns = stamp_to_nanoseconds(
                        point.header.stamp)
                    staging = bool(
                        self.detected_row == 1
                        and getattr(self, 'empty_arm_staging_enabled', False)
                    )
                    x, y, yaw = self._goal_from_target(
                        target,
                        self.shelf_normal,
                        (
                            getattr(self, 'top_row_grasp_standoff', self.grasp_standoff)
                            + (self.empty_arm_staging_backoff if staging else 0.0)
                            if self.detected_row == 1
                            else self.grasp_standoff
                        ),
                        self.grasp_lateral_bias,
                    )
                    self._empty_stage_pending = staging
                    if staging:
                        self._navigate(x, y, yaw, 'empty_arm_staging',
                                       profile='empty_arm_staging')
                    else:
                        self._navigate(x, y, yaw, 'book_grasp_standoff')
                    self._set_state('ALIGN_BOOK')
                    return
            if self._elapsed_state() > self.perception_timeout:
                self._abort('target_book_not_found')
            return

        if self.state == 'ALIGN_BOOK':
            if self._nav_reached():
                self.book_point = None
                if self.detected_row is None:
                    self._abort('confirmed_book_row_missing')
                    return
                # Stop detections while the camera moves so a transient point
                # cannot be accepted as the close-range grasp target.
                self._perception_mode('idle')
                command = f'look_book_row_{self.detected_row}'
                self._manipulate(command)
                self._set_state('HEAD_REACQUIRE_BOOK', command=command)
            elif self._nav_failed() or self._elapsed_state() > self.navigation_timeout:
                self._abort('book_alignment_failed')
            return

        if self.state == 'HEAD_REACQUIRE_BOOK':
            command = f'look_book_row_{self.detected_row}'
            if self._manip_succeeded(command):
                self.book_point = None
                self._book_reacquire_after_ns = self.get_clock().now().nanoseconds
                if self._perception_mode('books') is False:
                    return
                self._set_state('REACQUIRE_BOOK')
            elif self._manip_failed() or self._elapsed_state() > self.manipulation_timeout:
                self._abort('book_reacquisition_head_pose_failed')
            return

        if self.state == 'REACQUIRE_BOOK':
            if self.book_point is not None:
                if (stamp_to_nanoseconds(self.book_point.header.stamp)
                        <= getattr(self, '_book_reacquire_after_ns', 0)):
                    self.book_point = None
                    return
                if getattr(self, '_empty_stage_pending', False):
                    try:
                        point = self.tf_buffer.transform(
                            self.book_point, 'odom', timeout=Duration(seconds=1.0),
                        )
                        from .empty_arm_preparation import point_in_pose
                        target = [point.point.x, point.point.y, point.point.z]
                        start = list(self.robot_pose)
                        front = point_in_pose(target, start)
                        advance = float(front[0]) - self.top_row_grasp_standoff
                        if not .04 <= advance <= .35:
                            raise ValueError('staging distance no longer matches the book')
                        goal = [start[0] + advance * math.cos(start[2]),
                                start[1] + advance * math.sin(start[2]), start[2]]
                    except (TransformException, ValueError, TypeError) as exc:
                        self._abort(f'empty_arm_target_invalid:{exc}')
                        return
                    self._empty_stage_plan_id = uuid.uuid4().hex
                    self._manipulate(
                        'prepare_pick_approach', plan_id=self._empty_stage_plan_id,
                        book_odom=target,
                        book_stamp_ns=stamp_to_nanoseconds(point.header.stamp),
                        staging_pose=start, final_goal=goal,
                    )
                    self._set_state('PREPARE_EMPTY_ARM')
                    return
                self.pick_attempts += 1
                # A retry may legitimately select another book of the target
                # colour.  Keep the previous exact identity only while that
                # payload might still be carried, then start each new pick
                # attempt with an unbound identity.
                self.target_book_model = None
                pick_payload = {}
                if getattr(self, 'lift_first_extraction_enabled', False):
                    try:
                        pick_payload['shelf_bay_context'] = make_bay_context(
                            self.target_marker_odom, self.shelf_normal,
                            self.target_physical_column,
                            self._shelf_registration_stamp_ns,
                            int(self.get_clock().now().nanoseconds),
                        )
                    except (ValueError, TypeError) as exc:
                        self._abort(f'lift_first_registration_invalid:{exc}')
                        return
                self._manipulate('pick', **pick_payload)
                self._set_state('PICK')
            elif self._elapsed_state() > self.perception_timeout:
                self._abort('book_reacquisition_failed')
            return

        if self.state == 'PREPARE_EMPTY_ARM':
            if self._manip_succeeded('prepare_pick_approach'):
                certificate = self.manip_event or {}
                age = (self.get_clock().now().nanoseconds
                       - int(certificate.get('prepared_stamp_ns', 0))) / 1e9
                if (certificate.get('plan_id') != self._empty_stage_plan_id
                        or not 0 <= age <= 1.0
                        or not certificate.get('prepared_joints')):
                    self._abort('empty_arm_preparation_certificate_invalid')
                    return
                goal = certificate['final_goal']
                self._navigate(
                    *goal, 'empty_arm_final_advance', profile='empty_arm_advance',
                    plan_id=self._empty_stage_plan_id,
                    prepared_stamp_ns=certificate['prepared_stamp_ns'],
                    prepared_joints=certificate['prepared_joints'],
                    advance_bounds=certificate['advance_bounds'],
                )
                self._empty_stage_pending = False
                # ALIGN_BOOK waits for navigation, then moves the head and
                # requires a fresh close-range observation before pick planning.
                self._set_state('ALIGN_BOOK')
            elif self._manip_failed() or self._elapsed_state() > self.manipulation_timeout:
                self._abort('empty_arm_preparation_failed')
            return

        if self.state == 'PICK':
            if self._manip_succeeded('pick'):
                if (
                    not bool(getattr(self, 'dry_run', False))
                    and not self._target_identity_confirmed()
                ):
                    self._abort('target_book_identity_not_confirmed')
                    return
                self._perception_mode('idle')
                try:
                    x, y, yaw = self._carried_shelf_retreat_goal()
                except ValueError as exc:
                    self._abort(f'carried_shelf_retreat_invalid:{exc}')
                else:
                    self._navigate(
                        x,
                        y,
                        yaw,
                        'carried_shelf_retreat',
                        profile='carried_retreat',
                    )
                    self._set_state('CLEAR_SHELF_WITH_BOOK')
            elif self._manip_failed():
                failure_reason = str((self.manip_event or {}).get('reason', ''))
                if failure_reason in ('pick_recovery_failed', 'pick_payload_lost'):
                    self._abort(failure_reason)
                elif self.pick_attempts < self.max_pick_attempts:
                    self.book_point = None
                    self._perception_mode('idle')
                    command = f'look_book_row_{self.detected_row}'
                    self._manipulate(command)
                    self._set_state('RETRY_HEAD_BOOKS')
                else:
                    self._abort('pick_attempts_exhausted')
            elif self._elapsed_state() > self.manipulation_timeout:
                self._abort('pick_command_timeout')
            return

        if self.state == 'CLEAR_SHELF_WITH_BOOK':
            if self._nav_reached():
                self._manipulate('compact_transport')
                self._set_state('COMPACT_TRANSPORT')
            elif self._nav_failed() or self._elapsed_state() > self.navigation_timeout:
                self._abort('carried_shelf_retreat_failed')
            return

        if self.state == 'COMPACT_TRANSPORT':
            if self._manip_succeeded('compact_transport'):
                start_x, start_y, _ = self.start_pose
                yaw = math.atan2(self.shelf_normal[1], self.shelf_normal[0])
                if getattr(self, 'head_return_overlap_enabled', False):
                    from .head_return_overlap import begin_mission
                    begin_mission(self, (start_x, start_y, yaw))
                else:
                    self._navigate(start_x, start_y, yaw, 'return_start_with_book')
                    self._set_state('RETURN_START')
            elif self._manip_failed() or self._elapsed_state() > self.manipulation_timeout:
                self._abort('carried_compaction_failed')
            return

        if self.state == 'RETRY_HEAD_BOOKS':
            command = f'look_book_row_{self.detected_row}'
            if self._manip_succeeded(command):
                self.book_point = None
                self._book_reacquire_after_ns = self.get_clock().now().nanoseconds
                if self._perception_mode('books') is False:
                    return
                self._set_state('REACQUIRE_BOOK')
            elif self._manip_failed() or self._elapsed_state() > self.manipulation_timeout:
                self._abort('pick_recovery_failed')
            return

        if self.state == 'RETURN_START':
            if getattr(self, '_head_return_request', None) is not None:
                from .head_return_overlap import tick_mission
                if tick_mission(self):
                    return
            if self._nav_reached():
                self._manipulate('look_bin')
                self._set_state('HEAD_BIN')
            elif self._nav_failed() or self._elapsed_state() > self.navigation_timeout:
                self._abort('return_navigation_failed')
            return

        if self.state == 'HEAD_BIN':
            if self._manip_succeeded('look_bin'):
                self.bin_point = None
                self._perception_mode('bin')
                self._set_state('FIND_BIN')
            elif self._manip_failed() or self._elapsed_state() > self.manipulation_timeout:
                self._abort('head_bin_pose_failed')
            return

        if self.state == 'FIND_BIN':
            if not bin_message_is_current(
                self.bin_point, self.get_clock().now().nanoseconds,
                getattr(self, 'bin_invalidated_ns', -1),
            ):
                self.bin_point = None
            if (
                getattr(self, 'bin_visibility_recovery_enabled', False)
                and not bin_message_is_verified(
                    self.bin_point, self.get_clock().now().nanoseconds,
                    getattr(self, 'bin_verified_ns', -1),
                    getattr(self, 'bin_invalidated_ns', -1),
                )
            ):
                self.bin_point = None
            if self.bin_point is not None:
                try:
                    point = self.tf_buffer.transform(
                        self.bin_point,
                        'odom',
                        timeout=Duration(seconds=1.0),
                    )
                    target = np.asarray(
                        [point.point.x, point.point.y, point.point.z]
                    )
                    x, y, yaw = self._goal_from_robot_side(
                        target, self.bin_standoff
                    )
                except (TransformException, ValueError) as exc:
                    self._log('bin_geometry_rejected', reason=str(exc))
                    self.bin_point = None
                else:
                    if getattr(self, 'bin_visibility_recovery_enabled', False):
                        self._bin_approach_reference = BinApproachReference(
                            tuple(float(value) for value in target),
                            stamp_to_nanoseconds(self.bin_point.header.stamp),
                            (x, y, yaw),
                        )
                        self._bin_visibility_attempts = 0
                        self._bin_visibility_started_ns = None
                    self.bin_point = None
                    self._perception_mode('idle')
                    self._navigate(x, y, yaw, 'bin_placement_standoff')
                    self._set_state('ALIGN_BIN')
                    return
            elif self._elapsed_state() > self.perception_timeout:
                self._abort('collection_bin_not_found')
            return

        if self.state == 'ALIGN_BIN':
            if self._nav_reached():
                self.bin_point = None
                self._perception_mode('idle')
                self._manipulate('look_bin')
                self._set_state('HEAD_REACQUIRE_BIN')
            elif self._nav_failed() or self._elapsed_state() > self.navigation_timeout:
                self._abort('bin_alignment_failed')
            return

        if self.state == 'HEAD_REACQUIRE_BIN':
            if self._manip_succeeded('look_bin'):
                self.bin_point = None
                if getattr(self, 'bin_visibility_recovery_enabled', False):
                    self.bin_candidate = None
                    self._bin_reacquire_after_ns = self.get_clock().now().nanoseconds
                self._perception_mode('bin')
                self._set_state('REACQUIRE_BIN')
            elif self._manip_failed() or self._elapsed_state() > self.manipulation_timeout:
                self._abort('bin_reacquisition_head_pose_failed')
            return

        if self.state == 'REACQUIRE_BIN':
            if not bin_message_is_current(
                self.bin_point, self.get_clock().now().nanoseconds,
                getattr(self, 'bin_invalidated_ns', -1),
            ):
                self.bin_point = None
            if (self.bin_point is not None
                    and getattr(self, 'bin_visibility_recovery_enabled', False)):
                try:
                    if not self._verify_bin_reacquisition():
                        self.bin_point = None
                except (TransformException, ValueError) as exc:
                    self._log('bin_reacquisition_rejected', reason=str(exc))
                    self._abort('bin_reacquisition_association_failed')
                    return
            if self.bin_point is not None:
                self._bin_visibility_started_ns = None
                # Keep bin production until manipulation captures its own fresh input.
                self._manipulate('place')
                self._set_state('PLACE')
            elif self._elapsed_state() > self.perception_timeout:
                if not self._try_bin_visibility_recovery():
                    self._abort('collection_bin_reacquisition_failed')
            return

        if self.state == 'BIN_VISIBILITY_RETREAT':
            if self._nav_reached():
                self.bin_point = self.bin_candidate = None
                self._perception_mode('idle')
                self._manipulate('look_bin')
                self._set_state('HEAD_REACQUIRE_BIN')
            elif self._nav_failed() or self._elapsed_state() > self.navigation_timeout:
                self._abort('bin_visibility_retreat_failed')
            return

        if self.state == 'PLACE':
            if self._manip_succeeded('place'):
                self._set_state('WAIT_BIN_CONTACT')
            elif self._manip_failed() or self._elapsed_state() > self.manipulation_timeout:
                self._abort('placement_failed')
            return

        if self.state == 'WAIT_BIN_CONTACT':
            if bool(getattr(self, 'dry_run', False)):
                # Dry-run manipulation intentionally does not move a physical
                # book.  Complete the state-machine exercise without inventing
                # either scoped-model or collection-bin contact evidence.
                self._complete()
            elif getattr(self, 'delivery_evidence_enabled', False):
                evidence = self.delivery_release_evidence
                snapshot = None if evidence is None else evidence.evaluate(
                    int(self.get_clock().now().nanoseconds), time.monotonic())
                self._delivery_release_snapshot = snapshot
                if snapshot and snapshot['operation_completed']:
                    self._complete()
                elif not snapshot or snapshot['status'] in ('invalid', 'expired'):
                    self._abort('placement_operation_unverified:' +
                                str((snapshot or {}).get('reason', 'missing_attempt')))
            elif self.bin_contact_confirmed:
                self._complete()
            elif self._elapsed_state() > 4.0:
                self._abort('target_book_bin_contact_not_confirmed')

    def _manip_succeeded(self, command: str) -> bool:
        return bool(
            self.manip_event
            and self.manip_event.get('event') == 'succeeded'
            and self.manip_event.get('command') == command
        )

    def _manip_failed(self) -> bool:
        return bool(
            self.manip_event
            and self.manip_event.get('event') in ('failed', 'rejected', 'cancelled')
        )

    def _nav_reached(self) -> bool:
        return bool(self.nav_event and self.nav_event.get('event') == 'reached')

    def _nav_failed(self) -> bool:
        return bool(
            self.nav_event
            and self.nav_event.get('event') in ('failed', 'rejected', 'cancelled')
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        if not node.finished:
            try:
                if rclpy.ok():
                    node._abort('shutdown')
                else:
                    node._finalize_without_ros('shutdown')
            except Exception:
                # SIGINT can invalidate individual publishers just before
                # rclpy.ok() reflects the context transition on Humble.
                node._finalize_without_ros('shutdown')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
