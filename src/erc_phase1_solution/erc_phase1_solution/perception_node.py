"""Live RGB-D perception node and timestamped scoring-evidence recorder."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
from typing import Deque, Optional, Sequence, Tuple

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, ChannelFloat32, Image, PointCloud
from geometry_msgs.msg import Point32, PointStamped
from std_msgs.msg import Int32, String
from tf2_ros import Buffer, TransformException, TransformListener

from .bin_geometry import (
    BIN_MAX_SKEW_NS,
    BinTracker,
    bin_stamp_is_fresh,
    bin_surface_camera_point,
    camera_transform,
    depth_in_metres,
    red_bin_candidates,
    verify_bin_candidate,
)

from .common import RELIABLE_QOS, TRANSIENT_RELIABLE_QOS, decode_event, encode_event
from .book_selection_context import decode_context, select_registered_book
from .table_scene import fit_table_scene
from .bin_scene import fit_bin_scene
from .runtime_utils import (
    frames_are_synchronized,
    stamp_to_nanoseconds,
)
from .target_tracking import (
    TargetBookObservation,
    TargetBookTracker,
    TargetTrackingResult,
    estimate_target_book_observation,
)
from .vision import (
    MarkerDetection,
    annotate_bin,
    annotate_books,
    annotate_markers,
    deproject_pixel,
    detect_colored_books,
    detect_number_markers,
    detect_shelf_books,
    has_complete_book_row_layout,
    load_digit_templates,
    median_valid_depth,
    resolve_book_row,
)


class PerceptionNode(Node):
    """Detect requested marker/book/bin using only the live onboard RGB-D feed."""

    def __init__(self) -> None:
        super().__init__('erc_perception')
        self._declare_parameters()
        self.target_column = int(self.get_parameter('shelf_column_number').value)
        self.target_colour = str(self.get_parameter('book_colour').value).lower()
        self.confirmed_book_row: Optional[int] = None
        self.book_selection_context = None
        self.book_frame_pairs = deque(maxlen=6)
        self.stability_frames = int(
            self.get_parameter('detection_stability_frames').value
        )
        self.marker_confidence = float(
            self.get_parameter('minimum_marker_confidence').value
        )
        self.minimum_colour_area = float(
            self.get_parameter('minimum_colour_area_px').value
        )
        self.process_period = 1.0 / max(
            10.0, float(self.get_parameter('publish_rate_hz').value)
        )
        self.maximum_frame_age = float(
            self.get_parameter('maximum_frame_age_seconds').value
        )
        self.maximum_frame_skew = float(
            self.get_parameter('maximum_rgb_depth_skew_seconds').value
        )
        self.maximum_tracking_frame_skew = float(
            self.get_parameter(
                'target_tracking_maximum_frame_skew_seconds'
            ).value
        )
        self.tracking_maximum_age = float(
            self.get_parameter('target_tracking_maximum_age_seconds').value
        )
        self.tracking_minimum_depth_coverage = float(
            self.get_parameter('target_tracking_minimum_depth_coverage').value
        )
        self.tracking_maximum_plane_residual = float(
            self.get_parameter(
                'target_tracking_maximum_plane_residual_m'
            ).value
        )
        self.tracking_maximum_center_uncertainty = float(
            self.get_parameter(
                'target_tracking_maximum_center_uncertainty_m'
            ).value
        )
        self.tracking_maximum_orientation_uncertainty = float(
            self.get_parameter(
                'target_tracking_maximum_orientation_uncertainty_rad'
            ).value
        )
        self.target_tracker = TargetBookTracker(
            minimum_rate_hz=float(
                self.get_parameter('target_tracking_minimum_rate_hz').value
            ),
            minimum_consecutive_observations=int(
                self.get_parameter(
                    'target_tracking_minimum_consecutive_frames'
                ).value
            ),
            maximum_age_seconds=self.tracking_maximum_age,
            maximum_reacquisition_gap_seconds=float(
                self.get_parameter(
                    'target_tracking_maximum_reacquisition_gap_seconds'
                ).value
            ),
            maximum_center_speed_mps=float(
                self.get_parameter(
                    'target_tracking_maximum_center_speed_mps'
                ).value
            ),
            maximum_center_uncertainty_m=(
                self.tracking_maximum_center_uncertainty
            ),
        )
        self.template_dir = Path(str(self.get_parameter('digit_template_dir').value))
        self.templates = load_digit_templates(self.template_dir)
        self.output_dir = Path(str(self.get_parameter('image_output_dir').value))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge()
        self.mode = 'idle'
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_rgb_message: Optional[Image] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_message: Optional[Image] = None
        self.camera_info: Optional[CameraInfo] = None
        self.last_processed_ns = 0
        self.last_tracking_depth_ns = -1
        self.last_waiting_report_wall = 0.0
        self.ready_sent = False
        self.marker_history: Deque[Tuple[int, int, int]] = deque(
            maxlen=self.stability_frames
        )
        self.book_history: Deque[Tuple[int, int, int]] = deque(
            maxlen=self.stability_frames
        )
        self.bin_history: Deque[Tuple[int, int]] = deque(maxlen=self.stability_frames)
        self.bin_tracker = BinTracker(self.stability_frames)
        self.bin_rgb_frames = deque(maxlen=24)
        self.bin_depth_frames = deque(maxlen=24)
        self.last_bin_consumed_ns = -1
        self.last_bin_depth_ns = -1
        self.last_bin_verified_ns = -1
        self.last_bin_observed_ns = -1
        self.bin_table_height = float(self.get_parameter('bin_table_height').value)
        self.table_scene_enabled = bool(self.get_parameter('table_scene_enabled').value)
        self.bin_scene_enabled = bool(self.get_parameter('bin_scene_enabled').value)
        if self.bin_scene_enabled and not self.table_scene_enabled:
            raise ValueError('bin_scene_enabled requires table_scene_enabled')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.saved_modes = set()
        self.last_tracking_status = ''

        rgb_topic = str(self.get_parameter('rgb_topic').value)
        depth_topic = str(self.get_parameter('depth_topic').value)
        info_topic = str(self.get_parameter('camera_info_topic').value)
        # The official ros_gz bridge publishes the RGB-D streams as reliable.
        # Matching that QoS avoids fragmented image loss in Cyclone DDS.
        self.create_subscription(Image, rgb_topic, self._on_rgb, RELIABLE_QOS)
        self.create_subscription(Image, depth_topic, self._on_depth, RELIABLE_QOS)
        self.create_subscription(CameraInfo, info_topic, self._on_info, RELIABLE_QOS)
        self.create_subscription(
            String,
            '/erc/perception/mode',
            self._on_mode,
            TRANSIENT_RELIABLE_QOS,
        )
        self.marker_pub = self.create_publisher(
            PointCloud, '/erc/perception/markers', TRANSIENT_RELIABLE_QOS
        )
        self.book_pub = self.create_publisher(
            PointStamped, '/erc/perception/target_book', TRANSIENT_RELIABLE_QOS
        )
        # Unlike the mission's latched coarse point, this stream is volatile:
        # a new manipulation consumer must never receive an old retained pose.
        self.target_tracking_pub = self.create_publisher(
            PointCloud, '/erc/perception/target_book_observation', RELIABLE_QOS
        )
        self.target_tracking_status_pub = self.create_publisher(
            String, '/erc/perception/target_book_tracking_status', RELIABLE_QOS
        )
        self.bin_pub = self.create_publisher(
            PointStamped, '/erc/perception/bin', TRANSIENT_RELIABLE_QOS
        )
        self.row_pub = self.create_publisher(
            Int32, '/erc/perception/row', TRANSIENT_RELIABLE_QOS
        )
        self.status_pub = self.create_publisher(
            String, '/erc/perception/status', TRANSIENT_RELIABLE_QOS
        )
        self.create_timer(self.process_period, self._process)

    def _declare_parameters(self) -> None:
        values = {
            'shelf_column_number': 1,
            'book_colour': 'red',
            'digit_template_dir': '',
            'image_output_dir': 'erc_images',
            'rgb_topic': '/head_front_camera/head_front_camera/color/image_raw',
            'depth_topic': '/head_front_camera/head_front_camera/depth/image_rect_raw',
            'camera_info_topic': '/head_front_camera/head_front_camera/depth/camera_info',
            'detection_stability_frames': 3,
            'minimum_marker_confidence': 0.48,
            'minimum_colour_area_px': 70.0,
            'publish_rate_hz': 15.0,
            'maximum_frame_age_seconds': 1.0,
            'maximum_rgb_depth_skew_seconds': 0.20,
            'bin_table_height': 0.73,
            'table_scene_enabled': False,
            'bin_scene_enabled': False,
            'target_tracking_minimum_rate_hz': 10.0,
            'target_tracking_maximum_age_seconds': 0.15,
            'target_tracking_maximum_frame_skew_seconds': 0.04,
            'target_tracking_minimum_consecutive_frames': 3,
            'target_tracking_maximum_reacquisition_gap_seconds': 0.25,
            'target_tracking_maximum_center_speed_mps': 0.45,
            'target_tracking_minimum_depth_coverage': 0.65,
            'target_tracking_maximum_plane_residual_m': 0.002,
            'target_tracking_maximum_center_uncertainty_m': 0.0008,
            'target_tracking_maximum_orientation_uncertainty_rad': 0.05,
        }
        for name, default in values.items():
            self.declare_parameter(name, default)

    def _publish_status(self, event: str, **fields) -> None:
        message = String()
        message.data = encode_event(event, mode=self.mode, **fields)
        self.status_pub.publish(message)

    def _on_rgb(self, message: Image) -> None:
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            self.latest_rgb_message = message
            if self.mode == 'bin':
                self.bin_rgb_frames.append((message, self.latest_rgb))
        except CvBridgeError as exc:
            self._publish_status('frame_error', sensor='rgb', reason=str(exc))

    def _on_depth(self, message: Image) -> None:
        try:
            self.latest_depth = np.asarray(
                self.bridge.imgmsg_to_cv2(message, desired_encoding='passthrough')
            )
            self.latest_depth_message = message
            if self.mode == 'bin':
                self.bin_depth_frames.append((message, self.latest_depth))
        except CvBridgeError as exc:
            self._publish_status('frame_error', sensor='depth', reason=str(exc))

    def _on_info(self, message: CameraInfo) -> None:
        self.camera_info = message

    def _on_mode(self, message: String) -> None:
        payload = decode_event(message.data)
        mode = str(payload.get('event', message.data)).strip().lower()
        if mode not in ('idle', 'markers', 'books', 'bin'):
            self._publish_status('mode_rejected', requested=mode)
            return
        from .empty_head_timing import camera_mode
        try:
            camera_mode(self, payload)
        except RuntimeError as error:
            self._publish_status('mode_rejected', requested=mode, reason=str(error))
            return
        if 'head_return_not_before_ns' in payload:
            from .head_return_overlap import camera_epoch
            try:
                camera_epoch(self, payload)
            except RuntimeError as error:
                self._publish_status('mode_rejected', requested=mode, reason=str(error))
                return
        previous_column = self.target_column
        previous_colour = self.target_colour
        previous_context = getattr(self, 'book_selection_context', None)
        if 'shelf_column_number' in payload:
            value = int(payload['shelf_column_number'])
            if 1 <= value <= 5:
                self.target_column = value
        if 'book_colour' in payload:
            value = str(payload['book_colour']).lower()
            if value in ('red', 'blue', 'green', 'yellow'):
                self.target_colour = value
        previous_confirmed_row = self.confirmed_book_row
        raw_confirmed_row = payload.get('confirmed_row') if mode == 'books' else None
        if raw_confirmed_row is None:
            self.confirmed_book_row = None
        else:
            value = int(raw_confirmed_row)
            self.confirmed_book_row = value if 1 <= value <= 4 else None
        self.book_selection_context = None
        if mode == 'books':
            try:
                self.book_selection_context = decode_context(
                    payload.get('book_selection_context'),
                    int(self.get_clock().now().nanoseconds),
                    column=self.target_column, colour=self.target_colour,
                    confirmed_row=self.confirmed_book_row)
            except (TypeError, ValueError) as error:
                # A rejected replacement cannot leave the previous identity
                # publishing from an older valid mode/context.
                self.mode = 'idle'
                self.target_tracker.reset()
                self.book_history.clear()
                if hasattr(self, 'book_frame_pairs'):
                    self.book_frame_pairs.clear()
                self.last_tracking_depth_ns = -1
                self._publish_tracking_status('target_tracking_unavailable',
                    force=True, reason=str(error))
                self._publish_status('mode_rejected', requested=mode, reason=str(error))
                return
        target_changed = (
            self.target_column != previous_column
            or self.target_colour != previous_colour
            or self.confirmed_book_row != previous_confirmed_row
            or self.book_selection_context != previous_context
        )
        if mode != self.mode or target_changed:
            self.mode = mode
            self.marker_history.clear()
            self.book_history.clear()
            if hasattr(self, 'book_frame_pairs'):
                self.book_frame_pairs.clear()
            self.bin_history.clear()
            self.bin_tracker.reset()
            self.bin_rgb_frames.clear()
            self.bin_depth_frames.clear()
            self.last_bin_consumed_ns = -1
            self.last_bin_depth_ns = -1
            self.last_bin_verified_ns = -1
            self.last_bin_observed_ns = -1
            self.target_tracker.reset()
            self.last_tracking_depth_ns = -1
            self.last_tracking_status = ''
            self._publish_tracking_status(
                'target_tracking_unavailable',
                force=True,
                reason=(
                    'target_not_observed'
                    if mode == 'books'
                    else 'perception_mode_not_books'
                ),
            )
            self._publish_status('mode_changed', target_column=self.target_column,
                                 target_colour=self.target_colour)
        if ('marker_search_request_id' in payload
                or getattr(self, '_marker_search_request', None) is not None):
            from .marker_search_completion import receive_request
            receive_request(self, payload)
        from .place_input_handoff import perception_idle
        perception_idle(self, payload)

    def _ready(self) -> bool:
        from .empty_head_timing import camera_ready
        if not camera_ready(self):
            return False
        available = (
            self.latest_rgb is not None
            and self.latest_rgb_message is not None
            and self.latest_depth is not None
            and self.latest_depth_message is not None
            and self.camera_info is not None
            and self.camera_info.k[0] > 0.0
            and self.camera_info.k[4] > 0.0
        )
        if not available:
            return False
        if self.latest_rgb.shape[:2] != self.latest_depth.shape[:2]:
            return False
        if not frames_are_synchronized(
            self.latest_rgb_message.header.stamp,
            self.latest_depth_message.header.stamp,
            self.maximum_frame_skew,
        ):
            return False
        newest_stamp = max(
            stamp_to_nanoseconds(self.latest_rgb_message.header.stamp),
            stamp_to_nanoseconds(self.latest_depth_message.header.stamp),
        )
        age = (self.get_clock().now().nanoseconds - newest_stamp) / 1e9
        return -0.10 <= age <= self.maximum_frame_age

    @staticmethod
    def _stable(history: Sequence[Tuple[int, ...]], tolerance_px: int = 16) -> bool:
        if len(history) == 0 or len(history) < history.maxlen:  # type: ignore[attr-defined]
            return False
        array = np.asarray(history, dtype=float)
        if array.shape[1] > 2 and not np.all(array[:, 2] == array[0, 2]):
            return False
        return bool(np.max(np.ptp(array[:, :2], axis=0)) <= tolerance_px)

    def _depth_scale(self) -> float:
        if self.latest_depth is not None and self.latest_depth.dtype.kind in 'ui':
            return 0.001
        return 1.0

    def _deproject(self, pixel: Tuple[float, float], radius: int = 4):
        depth = median_valid_depth(
            self.latest_depth,
            pixel,
            radius=radius,
            min_depth=0.20,
            max_depth=8.0,
            depth_scale=self._depth_scale(),
        )
        if depth is None:
            return None
        return deproject_pixel(pixel, depth, self.camera_info)

    def _header(self):
        return self.latest_depth_message.header

    def latest_target_observation(
        self,
        *,
        maximum_age_seconds: Optional[float] = None,
    ) -> TargetTrackingResult:
        """Return a manipulation-safe latest target observation or a reason.

        This in-process API and the volatile
        ``/erc/perception/target_book_observation`` topic expose the same
        guarded state.  The method never falls back to the coarse, latched
        mission point when the live target is stale or occluded.
        """

        return self.target_tracker.latest(
            self.get_clock().now().nanoseconds,
            maximum_age_seconds=maximum_age_seconds,
        )

    def _publish_tracking_status(
        self,
        event: str,
        *,
        force: bool = False,
        **fields,
    ) -> None:
        # Sequence, timestamp, and numeric quality change every frame.  The
        # point cloud already carries them; status is edge-triggered so a
        # reliable subscriber is not flooded at camera rate.
        signature = encode_event(
            event,
            **{
                name: fields[name]
                for name in (
                    'reason',
                    'track_id',
                    'track_generation',
                    'target_colour',
                    'row',
                )
                if name in fields
            },
        )
        if not force and signature == self.last_tracking_status:
            return
        self.last_tracking_status = signature
        message = String()
        message.data = encode_event(event, mode=self.mode, **fields)
        self.target_tracking_status_pub.publish(message)

    def _tracking_unavailable(
        self,
        stamp_ns: int,
        reason: str,
        *,
        clear_book_history: bool = True,
        **metrics,
    ) -> None:
        self.target_tracker.mark_unavailable(stamp_ns, reason)
        if clear_book_history:
            self.book_history.clear()
        self._publish_tracking_status(
            'target_tracking_unavailable',
            reason=reason,
            stamp_ns=int(stamp_ns),
            **metrics,
        )

    def _publish_target_tracking(
        self,
        observation: TargetBookObservation,
    ) -> None:
        """Publish center + ordered face corners and repeated quality channels.

        Point role 0 is the centre; roles 1..4 are top-left, top-right,
        bottom-right, and bottom-left.  Repeating scalar metadata on all five
        points keeps the standard PointCloud representation unambiguous for
        consumers that validate channel lengths.
        """

        cloud = PointCloud()
        cloud.header.frame_id = observation.frame_id
        cloud.header.stamp.sec, cloud.header.stamp.nanosec = divmod(
            observation.stamp_ns, 1_000_000_000)
        points = (observation.center, *observation.corners)
        cloud.points = [
            Point32(x=float(point[0]), y=float(point[1]), z=float(point[2]))
            for point in points
        ]
        count = len(points)

        def channel(name: str, values) -> ChannelFloat32:
            result = ChannelFloat32(name=name)
            if isinstance(values, (int, float)):
                result.values = [float(values)] * count
            else:
                result.values = [float(value) for value in values]
            return result

        cloud.channels = [
            channel('point_role', range(count)),
            channel('target_row', observation.row),
            channel('track_generation', observation.track_generation),
            channel('sequence', observation.sequence),
            channel('detection_confidence', observation.detection_confidence),
            channel('quality_confidence', observation.quality_confidence),
            channel(
                'identity_continuity_confidence',
                observation.identity_continuity_confidence,
            ),
            channel('depth_coverage', observation.depth_coverage),
            channel('plane_residual_m', observation.plane_residual_m),
            channel('center_uncertainty_m', observation.center_uncertainty_m),
            channel('extent_uncertainty_m', observation.extent_uncertainty_m),
            channel(
                'orientation_uncertainty_rad',
                observation.orientation_uncertainty_rad,
            ),
            channel('short_extent_m', observation.short_extent_m),
            channel('long_extent_m', observation.long_extent_m),
            channel('face_normal_x', observation.face_normal[0]),
            channel('face_normal_y', observation.face_normal[1]),
            channel('face_normal_z', observation.face_normal[2]),
            channel('long_axis_x', observation.long_axis[0]),
            channel('long_axis_y', observation.long_axis[1]),
            channel('long_axis_z', observation.long_axis[2]),
            channel('observed_rate_hz', observation.observed_rate_hz),
        ]
        self.target_tracking_pub.publish(cloud)

    def _process(self) -> None:
        if self.mode == 'bin':
            self._process_bin()
            return
        if not self._ready():
            if (self.mode == 'markers'
                    and getattr(self, '_marker_search_request', None) is not None):
                from .marker_search_completion import capture_frame, complete_frame
                complete_frame(self, capture_frame(self), 'frame_invalid', 0)
            if self.mode == 'books':
                self._tracking_unavailable(
                    self.get_clock().now().nanoseconds,
                    'rgbd_frames_unavailable_or_stale',
                )
            wall_now = time.monotonic()
            if wall_now - self.last_waiting_report_wall >= 2.0:
                self.last_waiting_report_wall = wall_now
                rgb_shape = (
                    list(self.latest_rgb.shape[:2])
                    if self.latest_rgb is not None
                    else None
                )
                depth_shape = (
                    list(self.latest_depth.shape[:2])
                    if self.latest_depth is not None
                    else None
                )
                skew = None
                age = None
                if self.latest_rgb_message is not None and self.latest_depth_message is not None:
                    rgb_ns = stamp_to_nanoseconds(self.latest_rgb_message.header.stamp)
                    depth_ns = stamp_to_nanoseconds(self.latest_depth_message.header.stamp)
                    skew = abs(rgb_ns - depth_ns) / 1e9
                    age = (self.get_clock().now().nanoseconds - max(rgb_ns, depth_ns)) / 1e9
                self._publish_status(
                    'waiting_for_frames',
                    rgb_shape=rgb_shape,
                    depth_shape=depth_shape,
                    camera_info=bool(
                        self.camera_info is not None
                        and self.camera_info.k[0] > 0.0
                        and self.camera_info.k[4] > 0.0
                    ),
                    skew_seconds=skew,
                    newest_frame_age_seconds=age,
                )
            return
        rgb_stamp_ns = stamp_to_nanoseconds(
            self.latest_rgb_message.header.stamp
        )
        # Books retain a few immutable RGB-D pairs for timestamped TF to catch
        # up.  Retry these on the next timer even without a new RGB callback.
        if self.mode == 'books':
            self._process_books()
            return
        if rgb_stamp_ns <= self.last_processed_ns:
            return
        self.last_processed_ns = rgb_stamp_ns
        if not self.ready_sent:
            self.ready_sent = True
            self._publish_status(
                'ready',
                rgb_width=int(self.latest_rgb.shape[1]),
                rgb_height=int(self.latest_rgb.shape[0]),
                depth_encoding=self.latest_depth_message.encoding,
            )
        if self.mode == 'markers':
            self._process_markers()
        elif self.mode == 'books':
            self._process_books()

    def _process_markers(self) -> None:
        marker_frame = None
        if getattr(self, '_marker_search_request', None) is not None:
            from .marker_search_completion import capture_frame, complete_frame
            marker_frame = capture_frame(self)
        try:
            detections = detect_number_markers(
                self.latest_rgb,
                self.templates,
                min_area=120.0,
                min_confidence=self.marker_confidence,
            )
        except Exception:
            if marker_frame is not None:
                complete_frame(self, marker_frame, 'processing_error', None)
            raise
        targets = [item for item in detections if item.digit == self.target_column]
        if not targets or len(detections) < 3:
            self.marker_history.clear()
            if marker_frame is not None:
                complete_frame(self, marker_frame,
                    'target_absent' if not targets else 'target_present_pending', len(targets))
            return
        target = max(targets, key=lambda item: item.confidence)
        self.marker_history.append(
            (int(round(target.center[0])), int(round(target.center[1])), target.digit)
        )
        if not self._stable(self.marker_history):
            if marker_frame is not None:
                complete_frame(self, marker_frame, 'target_present_pending', len(targets))
            return
        cloud = PointCloud()
        cloud.header = self._header()
        digit_channel = ChannelFloat32(name='digit')
        confidence_channel = ChannelFloat32(name='confidence')
        used_digits = set()
        selected = sorted(detections, key=lambda item: -item.confidence)
        for detection in selected:
            if detection.digit in used_digits:
                continue
            point = self._deproject(detection.center, radius=4)
            if point is None:
                continue
            used_digits.add(detection.digit)
            cloud.points.append(Point32(x=point[0], y=point[1], z=point[2]))
            digit_channel.values.append(float(detection.digit))
            confidence_channel.values.append(float(detection.confidence))
        if self.target_column not in used_digits or len(cloud.points) < 3:
            if marker_frame is not None:
                complete_frame(self, marker_frame, 'depth_rejected', len(targets))
            return
        cloud.channels = [digit_channel, confidence_channel]
        self.marker_pub.publish(cloud)
        if marker_frame is not None:
            complete_frame(self, marker_frame, 'cloud_published', len(targets))
        if 'markers' not in self.saved_modes:
            annotated = annotate_markers(self.latest_rgb, detections)
            self._draw_column_evidence(annotated, detections, target)
            path, digest = self._save_evidence('column', annotated)
            self.saved_modes.add('markers')
            self._publish_status(
                'markers_detected',
                target_digit=self.target_column,
                marker_count=len(cloud.points),
                confidence=target.confidence,
                evidence_path=str(path),
                sha256=digest,
            )

    def _draw_column_evidence(
        self,
        image: np.ndarray,
        detections: Sequence[MarkerDetection],
        target: MarkerDetection,
    ) -> None:
        centers = sorted(item.center[0] for item in detections)
        gaps = np.diff(centers)
        column_width = float(np.median(gaps)) if len(gaps) else target.bbox[2] * 2.5
        x1 = max(0, int(round(target.center[0] - column_width / 2.0)))
        x2 = min(image.shape[1] - 1, int(round(target.center[0] + column_width / 2.0)))
        y1 = max(0, target.bbox[1])
        y2 = min(image.shape[0] - 1, int(round(y1 + 2.45 * column_width)))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(
            image,
            f'TARGET SHELF COLUMN {self.target_column}',
            (max(4, x1), max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _registered_book_frame(self):
        context = getattr(self, 'book_selection_context', None)
        now_ns = int(self.get_clock().now().nanoseconds)
        if context is None:
            self._tracking_unavailable(now_ns, 'book_selection_context_missing_or_incompatible')
            return None
        if not hasattr(self, 'book_frame_pairs'):
            self.book_frame_pairs = deque(maxlen=6)
        pairs = self.book_frame_pairs
        latest = (self.latest_rgb_message, self.latest_rgb,
                  self.latest_depth_message, self.latest_depth)
        if not pairs or pairs[-1][0] is not latest[0] or pairs[-1][2] is not latest[2]:
            pairs.append(latest)
        reason = 'book_selection_tf_unavailable'
        for rgb_message, rgb, depth_message, depth in reversed(pairs):
            rgb_ns = stamp_to_nanoseconds(rgb_message.header.stamp)
            depth_ns = stamp_to_nanoseconds(depth_message.header.stamp)
            if depth_ns <= self.last_tracking_depth_ns:
                continue
            try:
                context.require_frame(rgb_ns, depth_ns, now_ns)
                if abs(rgb_ns-depth_ns)/1e9 > self.maximum_tracking_frame_skew:
                    reason = 'rgb_depth_skew_too_large_for_tracking'
                    continue
                transform = self.tf_buffer.lookup_transform(
                    'odom', depth_message.header.frame_id,
                    Time.from_msg(depth_message.header.stamp))
                matrix = camera_transform(
                    transform.transform.translation, transform.transform.rotation)
                if not np.all(np.isfinite(matrix)):
                    raise ValueError('book_selection_invalid_camera_transform')
            except TransformException:
                continue
            except (TypeError, ValueError) as error:
                reason = str(error)
                continue
            return rgb_message, rgb, depth_message, depth, matrix
        # Pending TF is not evidence of lost identity.  The same original
        # freshness deadline still invalidates a track if it cannot catch up.
        if now_ns-self.last_tracking_depth_ns > int(
                getattr(self, 'tracking_maximum_age', .15)*1e9):
            self._tracking_unavailable(now_ns, reason)
        return None

    def _process_books(self) -> None:
        selected_frame = self._registered_book_frame()
        if selected_frame is None:
            return
        rgb_message, rgb, depth_message, depth, odom_from_camera = selected_frame
        depth_stamp_ns = stamp_to_nanoseconds(
            depth_message.header.stamp
        )
        if depth_stamp_ns == self.last_tracking_depth_ns:
            return
        if depth_stamp_ns < self.last_tracking_depth_ns:
            self._tracking_unavailable(
                depth_stamp_ns,
                'out_of_order_depth_frame',
            )
            return
        self.last_tracking_depth_ns = depth_stamp_ns
        rgb_stamp_ns = stamp_to_nanoseconds(
            rgb_message.header.stamp
        )
        tracking_skew = abs(rgb_stamp_ns - depth_stamp_ns) / 1e9
        if tracking_skew > self.maximum_tracking_frame_skew:
            self._tracking_unavailable(
                depth_stamp_ns,
                'rgb_depth_skew_too_large_for_tracking',
                frame_skew_seconds=tracking_skew,
                maximum_frame_skew_seconds=self.maximum_tracking_frame_skew,
            )
            return
        books = detect_shelf_books(
            rgb,
            min_area=self.minimum_colour_area,
        )
        if (
            self.confirmed_book_row is None
            and not has_complete_book_row_layout(books)
        ):
            # During the overview pass, numbering a partial set top-to-bottom
            # can silently shift every row below a missed band.  Close-range
            # tracking is allowed to see fewer rows only after the mission has
            # explicitly carried forward a confirmed overview row.
            self._tracking_unavailable(
                depth_stamp_ns,
                'incomplete_book_row_layout',
                detected_book_count=len(books),
            )
            return
        context = getattr(self, 'book_selection_context', None)
        try:
            if context is None:
                raise ValueError('book_selection_context_missing_or_incompatible')
            context.require_frame(rgb_stamp_ns, depth_stamp_ns,
                                  int(self.get_clock().now().nanoseconds))
            def deproject_book(pixel):
                z = median_valid_depth(depth, pixel, radius=5,
                    min_depth=.20, max_depth=8.,
                    depth_scale=.001 if depth.dtype.kind in 'ui' else 1.)
                return None if z is None else deproject_pixel(pixel, z, self.camera_info)
            target, coarse_point = select_registered_book(
                books, context, deproject_book,
                odom_from_camera)
        except (TransformException, TypeError, ValueError) as error:
            self._tracking_unavailable(depth_stamp_ns,
                'book_selection_tf_unavailable' if isinstance(error, TransformException)
                else str(error))
            return
        row = resolve_book_row(target, self.confirmed_book_row)
        if row is None:
            self._tracking_unavailable(depth_stamp_ns, 'target_row_unresolved')
            return

        # The mission's initial shelf approach needs only a stable coarse
        # centre and row.  Do not make that navigation observation depend on
        # the stricter metric face tracker used to guard arm motion.  At the
        # overview distance an edge-on book can expose two same-colour faces;
        # the tracker may safely withhold its precision stream while the
        # median-depth coarse observation remains valid.
        self.book_history.append(
            (int(round(target.center[0])), int(round(target.center[1])), int(row))
        )
        stable_detection = self._stable(
            self.book_history,
            tolerance_px=12,
        )
        if self.confirmed_book_row is None and stable_detection:
            point = coarse_point
            if point is not None:
                message = PointStamped()
                message.header = depth_message.header
                message.point.x, message.point.y, message.point.z = point
                self.book_pub.publish(message)
                self.row_pub.publish(Int32(data=int(row)))
                if 'books' not in self.saved_modes:
                    annotated = annotate_books(
                        rgb,
                        books,
                        target=target,
                    )
                    path, digest = self._save_evidence('book', annotated)
                    self.saved_modes.add('books')
                    self._publish_status(
                        'book_detected',
                        target_colour=self.target_colour,
                        row=int(row),
                        confidence=target.confidence,
                        evidence_path=str(path),
                        sha256=digest,
                    )

        estimate = estimate_target_book_observation(
            rgb,
            depth,
            target,
            self.camera_info,
            stamp_ns=depth_stamp_ns,
            frame_id=str(depth_message.header.frame_id),
            row=int(row),
            depth_scale=.001 if depth.dtype.kind in 'ui' else 1.,
            minimum_depth_coverage=self.tracking_minimum_depth_coverage,
            maximum_plane_residual_m=self.tracking_maximum_plane_residual,
            maximum_center_uncertainty_m=(
                self.tracking_maximum_center_uncertainty
            ),
            maximum_orientation_uncertainty_rad=(
                self.tracking_maximum_orientation_uncertainty
            ),
        )
        if not estimate.ok or estimate.observation is None:
            self._tracking_unavailable(
                depth_stamp_ns,
                estimate.reason,
                clear_book_history=self.confirmed_book_row is not None,
                **estimate.metrics,
            )
            return
        measured_center = np.asarray(estimate.observation.center, dtype=float)
        measured_odom = (odom_from_camera[:3, :3] @ measured_center
                         + odom_from_camera[:3, 3])
        if not context.contains(measured_odom):
            self._tracking_unavailable(depth_stamp_ns,
                                       'book_selection_metric_center_outside_bay')
            return
        update = self.target_tracker.observe(estimate.observation)
        if not update.ok:
            if self.confirmed_book_row is not None:
                self.book_history.clear()
            self._publish_tracking_status(
                'target_tracking_unavailable',
                reason=update.reason,
                stamp_ns=depth_stamp_ns,
                **update.metrics,
            )
            return
        guarded = self.latest_target_observation()
        if not guarded.ok or guarded.observation is None:
            if (
                self.confirmed_book_row is not None
                and guarded.reason != 'target_track_warming_up'
            ):
                self.book_history.clear()
            self._publish_tracking_status(
                'target_tracking_unavailable',
                reason=guarded.reason,
                stamp_ns=depth_stamp_ns,
                **guarded.metrics,
            )
            return
        observation = guarded.observation
        self._publish_target_tracking(observation)
        self._publish_tracking_status(
            'target_tracking',
            track_id=observation.track_id,
            track_generation=observation.track_generation,
            target_colour=observation.target_colour,
            row=observation.row,
            stamp_ns=observation.stamp_ns,
            sequence=observation.sequence,
            observed_rate_hz=observation.observed_rate_hz,
            center_uncertainty_m=observation.center_uncertainty_m,
            orientation_uncertainty_rad=(
                observation.orientation_uncertainty_rad
            ),
            quality_confidence=observation.quality_confidence,
        )
        if self.confirmed_book_row is not None and stable_detection:
            message = PointStamped()
            message.header = depth_message.header
            message.point.x, message.point.y, message.point.z = (
                observation.center
            )
            self.book_pub.publish(message)
            self.row_pub.publish(Int32(data=int(row)))
            if 'books' not in self.saved_modes:
                annotated = annotate_books(
                    rgb,
                    books,
                    target=target,
                )
                path, digest = self._save_evidence('book', annotated)
                self.saved_modes.add('books')
                self._publish_status(
                    'book_detected',
                    target_colour=self.target_colour,
                    row=int(row),
                    confidence=target.confidence,
                    evidence_path=str(path),
                    sha256=digest,
                )
    def _invalidate_bin(self, reason: str, stamp_ns: int) -> None:
        self.bin_tracker.reset()
        self.last_bin_verified_ns = -1
        self.last_bin_observed_ns = -1
        self._publish_status(
            'bin_invalid', bin_valid=False, observation_stamp_ns=stamp_ns,
            reason=reason,
        )

    def _process_bin(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        selected = None
        if self.camera_info is not None and self.bin_depth_frames:
            # Keep recent frames while timestamped TF catches up with RGB-D.
            for rgb_message, rgb in reversed(self.bin_rgb_frames):
                stamp_ns = stamp_to_nanoseconds(rgb_message.header.stamp)
                if (stamp_ns <= self.last_bin_consumed_ns
                        or not bin_stamp_is_fresh(stamp_ns, now_ns)):
                    continue
                depth_message, depth = min(
                    self.bin_depth_frames,
                    key=lambda item: abs(stamp_to_nanoseconds(item[0].header.stamp)-stamp_ns),
                )
                depth_ns = stamp_to_nanoseconds(depth_message.header.stamp)
                head_epoch = getattr(self, '_head_return_camera_epoch_ns', None)
                if head_epoch is not None and (stamp_ns <= head_epoch or depth_ns <= head_epoch):
                    continue
                if (abs(depth_ns-stamp_ns) > BIN_MAX_SKEW_NS
                        or depth_ns <= getattr(self, 'last_bin_depth_ns', -1)
                        or not bin_stamp_is_fresh(depth_ns, now_ns)):
                    continue
                try:
                    transform = self.tf_buffer.lookup_transform(
                        'base_footprint', depth_message.header.frame_id,
                        Time.from_msg(depth_message.header.stamp),
                    )
                    matrix = camera_transform(
                        transform.transform.translation, transform.transform.rotation,
                    )
                except (TransformException, ValueError):
                    continue
                selected = rgb_message, rgb, depth_message, depth, matrix, stamp_ns, depth_ns
                break
        if selected is None:
            if not bin_stamp_is_fresh(getattr(self, 'last_bin_observed_ns', -1), now_ns):
                self._invalidate_bin('no_recent_synchronized_rgb_depth_tf', now_ns)
            return
        rgb_message, rgb, depth_message, depth, matrix, rgb_stamp_ns, stamp_ns = selected
        self.last_bin_consumed_ns = rgb_stamp_ns
        self.last_bin_depth_ns = stamp_ns
        candidates, rejected = [], []
        for detection in red_bin_candidates(rgb):
            observation, reason = verify_bin_candidate(
                rgb, depth, self.camera_info.k, matrix, detection, self.bin_table_height,
            )
            if observation is not None:
                point = bin_surface_camera_point(rgb, depth, self.camera_info.k, detection)
                if point is not None:
                    candidates.append((observation, point))
                else:
                    rejected.append('no_front_surface_depth')
            else:
                rejected.append(reason)
        if not candidates:
            self._invalidate_bin(','.join(sorted(set(rejected))) or 'no_red_candidate', stamp_ns)
            return
        observation, point = max(candidates, key=lambda item: item[0].detection.area)
        self.last_bin_observed_ns = stamp_ns
        if not self.bin_tracker.observe(observation, stamp_ns):
            self._publish_status('bin_checking', bin_valid=False, observation_stamp_ns=stamp_ns)
            return
        self.last_bin_verified_ns = stamp_ns
        message = PointStamped()
        # Coordinates use depth intrinsics: preserve its optical frame and
        # timestamp, even when a nearby RGB frame supplied the colour mask.
        message.header = depth_message.header
        message.point.x, message.point.y, message.point.z = map(float, point)
        scene_fields = {}
        if getattr(self, 'table_scene_enabled', False):
            bin_floor_base = matrix[:3, :3] @ np.asarray(point) + matrix[:3, 3]
            try:
                table_scene = fit_table_scene(rgb, depth, self.camera_info.k, matrix, bin_floor_base)
            except ValueError as exc:
                table_scene = dict(valid=False, reason=str(exc), frame='base_footprint')
            scene_fields = dict(table_scene=table_scene, bin_floor_point_base=bin_floor_base.tolist())
        if getattr(self, 'bin_scene_enabled', False):
            # Preserve the front-patch navigation point. Only placement uses
            # the independently fitted CAD center and orientation below.
            try:
                table = scene_fields.get('table_scene')
                if not isinstance(table, dict) or table.get('valid') is not True:
                    raise ValueError('matching_table_unavailable')
                registered_bin = fit_bin_scene(
                    rgb, depth_in_metres(depth), self.camera_info.k, matrix,
                    observation.detection, table_scene=table,
                )
                if registered_bin.get('valid') is True:
                    epoch_tf = self.tf_buffer.lookup_transform(
                        'odom', 'base_footprint', Time.from_msg(depth_message.header.stamp),
                    )
                    epoch_matrix = camera_transform(
                        epoch_tf.transform.translation, epoch_tf.transform.rotation,
                    )
                    scene_fields['scene_base_reference'] = dict(
                        frame_id=epoch_tf.header.frame_id,
                        child_frame_id=epoch_tf.child_frame_id,
                        requested_stamp_ns=stamp_ns,
                        producer_stamp_ns=stamp_to_nanoseconds(epoch_tf.header.stamp),
                        pose=[float(epoch_matrix[0, 3]), float(epoch_matrix[1, 3]),
                              float(np.arctan2(epoch_matrix[1, 0], epoch_matrix[0, 0]))],
                    )
            except (TransformException, ValueError, TypeError) as exc:
                registered_bin = dict(valid=False, frame='base_footprint', reason=str(exc))
            scene_fields['bin_scene'] = registered_bin
        self.bin_pub.publish(message)
        self._publish_status(
            'bin_verified', bin_valid=True, observation_stamp_ns=stamp_ns,
            **observation.metrics, **scene_fields,
        )
        if 'bin' not in self.saved_modes:
            annotated = annotate_bin(rgb, observation.detection)
            path, digest = self._save_evidence('bin', annotated, rgb_message=rgb_message)
            self.saved_modes.add('bin')
            self._publish_status(
                'bin_detected', confidence=observation.detection.confidence,
                evidence_path=str(path), sha256=digest,
            )

    def _save_evidence(self, kind: str, image: np.ndarray, *, rgb_message=None):
        wall = datetime.now(timezone.utc)
        wall_text = wall.isoformat(timespec='milliseconds')
        rgb_message = self.latest_rgb_message if rgb_message is None else rgb_message
        ros_ns = (
            int(rgb_message.header.stamp.sec) * 1_000_000_000
            + int(rgb_message.header.stamp.nanosec)
        )
        overlay = image.copy()
        label = f'{wall_text} | ROS {ros_ns} ns | LIVE RGB'
        cv2.rectangle(overlay, (0, overlay.shape[0] - 30),
                      (overlay.shape[1] - 1, overlay.shape[0] - 1), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            label,
            (8, overlay.shape[0] - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        stamp = wall.strftime('%Y%m%dT%H%M%S_%fZ')
        final_path = self.output_dir / f'{kind}_{stamp}_ros{ros_ns}.png'
        temporary = final_path.with_suffix('.tmp.png')
        if not cv2.imwrite(str(temporary), overlay):
            raise OSError(f'OpenCV could not write evidence image {temporary}')
        if cv2.imread(str(temporary), cv2.IMREAD_COLOR) is None:
            temporary.unlink(missing_ok=True)
            raise OSError(f'evidence image failed verification: {temporary}')
        temporary.replace(final_path)
        digest = hashlib.sha256(final_path.read_bytes()).hexdigest()
        return final_path, digest


def main(args=None) -> None:
    cv2.setNumThreads(1)
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # Humble can surface a low-level take_message RuntimeError while SIGINT
        # is invalidating the context.  Preserve genuine runtime failures.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
