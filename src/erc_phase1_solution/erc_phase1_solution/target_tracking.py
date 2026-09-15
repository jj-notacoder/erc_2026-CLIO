"""Fail-closed RGB-D tracking for the selected competition book.

This module is deliberately independent of ROS.  It turns one colour-book
detection and its registered depth image into a metric observation of the
visible book face, then maintains the temporal/identity guarantees needed by
guarded manipulation.  No simulator entity state or contact sensor is used.

The tracker distinguishes an accepted sample from a *usable* latest sample.
After startup, an occlusion, or a rate interruption it requires a short run of
fresh, geometrically consistent frames before :meth:`TargetBookTracker.latest`
returns success.  Consumers therefore cannot accidentally reuse the last good
pose while the target is hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np

from .vision import BBox, BookDetection, CameraIntrinsics, hsv_color_mask


Vector3 = tuple[float, float, float]
Quad3 = tuple[Vector3, Vector3, Vector3, Vector3]


@dataclass(frozen=True)
class TargetBookObservation:
    """One image/depth-derived estimate of the visible target-book face.

    Coordinates are in ``frame_id`` (normally the depth optical frame).
    ``corners`` are ordered top-left, top-right, bottom-right, bottom-left in
    the image.  The uncertainty fields are conservative one-sigma estimates;
    motion guards add the uncertainty of both observations to the measured
    displacement before approving a bound.
    """

    stamp_ns: int
    frame_id: str
    target_colour: str
    row: int
    center: Vector3
    corners: Quad3
    face_normal: Vector3
    long_axis: Vector3
    short_extent_m: float
    long_extent_m: float
    bbox: BBox
    image_center: tuple[float, float]
    detection_confidence: float
    depth_coverage: float
    plane_residual_m: float
    center_uncertainty_m: float
    extent_uncertainty_m: float
    orientation_uncertainty_rad: float
    quality_confidence: float
    track_id: str = ''
    track_generation: int = 0
    sequence: int = 0
    interframe_period_s: float = 0.0
    observed_rate_hz: float = 0.0
    identity_continuity_confidence: float = 0.0


@dataclass(frozen=True)
class TargetTrackingResult:
    """Result of estimation, tracking, freshness, and motion guards."""

    ok: bool
    reason: str
    observation: Optional[TargetBookObservation] = None
    metrics: Mapping[str, float] = field(default_factory=dict)


def _result(
    ok: bool,
    reason: str,
    observation: Optional[TargetBookObservation] = None,
    **metrics: float,
) -> TargetTrackingResult:
    return TargetTrackingResult(ok, reason, observation, dict(metrics))


def _intrinsics(value) -> CameraIntrinsics:
    if isinstance(value, CameraIntrinsics):
        result = value
    elif isinstance(value, Mapping):
        result = CameraIntrinsics(
            fx=float(value['fx']),
            fy=float(value['fy']),
            cx=float(value['cx']),
            cy=float(value['cy']),
        )
    elif hasattr(value, 'k') and len(value.k) >= 6:
        result = CameraIntrinsics(
            fx=float(value.k[0]),
            fy=float(value.k[4]),
            cx=float(value.k[2]),
            cy=float(value.k[5]),
        )
    else:
        raise TypeError(
            'intrinsics must be CameraIntrinsics, a mapping, '
            'or CameraInfo-like'
        )
    if not all(math.isfinite(item) for item in vars(result).values()):
        raise ValueError('camera intrinsics must be finite')
    if result.fx <= 0.0 or result.fy <= 0.0:
        raise ValueError('camera focal lengths must be positive')
    return result


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError('cannot normalize a degenerate vector')
    return np.asarray(vector, dtype=np.float64) / norm


def _ordered_quad(points: np.ndarray) -> np.ndarray:
    """Order image points top-left, top-right, bottom-right, bottom-left."""

    points = np.asarray(points, dtype=np.float64).reshape(4, 2)
    order_y = np.argsort(points[:, 1], kind='stable')
    top = points[order_y[:2]]
    bottom = points[order_y[2:]]
    top = top[np.argsort(top[:, 0], kind='stable')]
    bottom = bottom[np.argsort(bottom[:, 0], kind='stable')]
    return np.asarray((top[0], top[1], bottom[1], bottom[0]))


def _fit_plane(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit and robustly refine a plane, returning centre, normal, residuals."""

    if len(points) < 12:
        raise ValueError('too few points for a stable plane')
    working = np.asarray(points, dtype=np.float64)
    for _ in range(2):
        center = np.mean(working, axis=0)
        _, singular, vectors = np.linalg.svd(
            working - center, full_matrices=False
        )
        if len(singular) < 3 or singular[1] <= 1e-8:
            raise ValueError('depth support is geometrically degenerate')
        normal = _normalize(vectors[-1])
        all_residuals = np.abs((points - center) @ normal)
        median = float(np.median(all_residuals))
        sigma = 1.4826 * float(np.median(np.abs(all_residuals - median)))
        threshold = max(0.00035, median + 4.5 * max(sigma, 1e-6))
        inliers = all_residuals <= threshold
        if int(np.count_nonzero(inliers)) < 12:
            raise ValueError('too few planar depth inliers')
        working = points[inliers]
    center = np.mean(working, axis=0)
    _, singular, vectors = np.linalg.svd(working - center, full_matrices=False)
    if singular[1] <= 1e-8:
        raise ValueError('depth support is geometrically degenerate')
    normal = _normalize(vectors[-1])
    # Camera optical coordinates look along +Z.  Give the normal a stable sign:
    # it points from the visible face back toward the camera.
    if float(np.dot(normal, center)) > 0.0:
        normal = -normal
    residuals = np.abs((working - center) @ normal)
    return center, normal, residuals


def _dominant_front_plane(
    points: np.ndarray,
    image_u: np.ndarray,
    image_v: np.ndarray,
    *,
    maximum_fit_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select the dominant camera-facing face of a uniformly coloured book.

    The competition books use one material on every face.  Near the centre of
    a shelf column, their colour component therefore contains both the narrow
    front spine and a receding side face.  A single PCA plane through that
    folded surface can be centimetres behind the spine.  Generate a bounded,
    deterministic set of local plane hypotheses and retain only a spatially
    broad consensus belonging to the camera-facing face.

    The returned Boolean mask indexes ``points`` and is also the authoritative
    image support for the face rectangle.  This prevents rejected side-face
    pixels from biasing the reported centre and extents.
    """

    points = np.asarray(points, dtype=np.float64)
    image_u = np.asarray(image_u, dtype=np.int64)
    image_v = np.asarray(image_v, dtype=np.int64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or image_u.shape != (len(points),)
        or image_v.shape != (len(points),)
    ):
        raise ValueError('plane support arrays are inconsistent')
    if len(points) < 60:
        raise ValueError('too few points for dominant plane support')

    score_indices = np.arange(len(points), dtype=np.int64)
    if len(score_indices) > maximum_fit_points:
        score_indices = np.linspace(
            0,
            len(score_indices) - 1,
            maximum_fit_points,
            dtype=np.int64,
        )
    score_points = points[score_indices]

    minimum_u, maximum_u = int(np.min(image_u)), int(np.max(image_u))
    minimum_v, maximum_v = int(np.min(image_v)), int(np.max(image_v))
    valid_image_mask = np.zeros(
        (maximum_v - minimum_v + 1, maximum_u - minimum_u + 1),
        dtype=np.uint8,
    )
    valid_image_mask[image_v - minimum_v, image_u - minimum_u] = 1
    interior_mask = cv2.erode(
        valid_image_mask,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    interior_v, interior_u = np.nonzero(interior_mask)
    anchors = [
        (int(u + minimum_u), int(v + minimum_v))
        for v, u in zip(interior_v, interior_u)
    ]
    point_indices = {
        (int(u), int(v)): index
        for index, (u, v) in enumerate(zip(image_u, image_v))
    }

    triples: list[tuple[int, int, int]] = []
    anchor_set = set(anchors)
    for gap in (2, 4):
        for u, v in anchors:
            right = (u + gap, v)
            below = (u, v + gap)
            if right in anchor_set and below in anchor_set:
                triples.append(
                    (
                        point_indices[(u, v)],
                        point_indices[right],
                        point_indices[below],
                    )
                )
    if len(triples) > 128:
        hypothesis_indices = np.linspace(
            0,
            len(triples) - 1,
            128,
            dtype=np.int64,
        )
        triples = [triples[int(index)] for index in hypothesis_indices]

    hypotheses: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        global_center, global_normal, _ = _fit_plane(score_points)
    except (ValueError, np.linalg.LinAlgError):
        pass
    else:
        hypotheses.append((global_center, global_normal))
    for first, second, third in triples:
        anchor = points[first]
        normal = np.cross(points[second] - anchor, points[third] - anchor)
        magnitude = float(np.linalg.norm(normal))
        if math.isfinite(magnitude) and magnitude > 1e-10:
            hypotheses.append((anchor, normal / magnitude))
    if not hypotheses:
        raise ValueError('no non-degenerate plane hypotheses')

    seed_tolerance_m = 0.00075
    refine_tolerance_m = 0.0010
    required_score_support = max(
        60,
        int(math.ceil(0.60 * len(score_points))),
    )
    score_u = image_u[score_indices]
    score_v = image_v[score_indices]
    original_score_span = (
        int(np.ptp(score_u)),
        int(np.ptp(score_v)),
    )
    best = None
    for anchor, hypothesis_normal in hypotheses:
        seed_residuals = np.abs(
            (score_points - anchor) @ hypothesis_normal
        )
        seed_inliers = seed_residuals <= seed_tolerance_m
        if int(np.count_nonzero(seed_inliers)) < required_score_support:
            continue
        try:
            center, normal, _ = _fit_plane(score_points[seed_inliers])
        except (ValueError, np.linalg.LinAlgError):
            continue
        refined_residuals = np.abs((score_points - center) @ normal)
        refined_inliers = refined_residuals <= refine_tolerance_m
        support_count = int(np.count_nonzero(refined_inliers))
        if support_count < required_score_support:
            continue
        support = score_points[refined_inliers]
        candidate_span = (
            int(np.ptp(score_u[refined_inliers])),
            int(np.ptp(score_v[refined_inliers])),
        )
        if (
            min(candidate_span) < 3
            or max(candidate_span) < 0.70 * max(original_score_span)
        ):
            continue
        support_center = np.mean(support, axis=0)
        try:
            camera_facing = abs(
                float(np.dot(normal, _normalize(support_center)))
            )
        except ValueError:
            continue
        if camera_facing < 0.80:
            continue
        support_rms = float(
            np.sqrt(np.mean(np.square(refined_residuals[refined_inliers])))
        )
        median_depth = float(np.median(support[:, 2]))
        score = (
            support_count,
            camera_facing,
            -support_rms,
            -median_depth,
        )
        if best is None or score > best[0]:
            best = (score, center, normal)
    if best is None:
        raise ValueError('no dominant camera-facing plane support')

    _, plane_center, normal = best
    full_inliers = (
        np.abs((points - plane_center) @ normal) <= refine_tolerance_m
    )
    required_full_support = max(60, int(math.ceil(0.60 * len(points))))
    if int(np.count_nonzero(full_inliers)) < required_full_support:
        raise ValueError('dominant plane support is too small')

    support_u = image_u[full_inliers]
    support_v = image_v[full_inliers]
    original_span = (
        int(np.ptp(image_u)),
        int(np.ptp(image_v)),
    )
    support_span = (
        int(np.ptp(support_u)),
        int(np.ptp(support_v)),
    )
    if (
        min(support_span) < 3
        or max(support_span) < 0.70 * max(original_span)
    ):
        raise ValueError('dominant plane support is spatially incomplete')

    support_points = points[full_inliers]
    if len(support_points) > maximum_fit_points:
        sample = np.linspace(
            0,
            len(support_points) - 1,
            maximum_fit_points,
            dtype=np.int64,
        )
        support_points = support_points[sample]
    plane_center, normal, residuals = _fit_plane(support_points)
    return plane_center, normal, residuals, full_inliers


def _point_on_plane(
    pixel: Sequence[float],
    plane_center: np.ndarray,
    normal: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    u, v = map(float, pixel)
    ray = np.asarray(
        (
            (u - intrinsics.cx) / intrinsics.fx,
            (v - intrinsics.cy) / intrinsics.fy,
            1.0,
        ),
        dtype=np.float64,
    )
    denominator = float(np.dot(normal, ray))
    if abs(denominator) <= 1e-8:
        raise ValueError('book plane is parallel to a camera ray')
    distance = float(np.dot(normal, plane_center)) / denominator
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError('book plane intersects behind the camera')
    return ray * distance


def estimate_target_book_observation(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    detection: BookDetection,
    intrinsics,
    *,
    stamp_ns: int,
    frame_id: str,
    row: int,
    depth_scale: float = 1.0,
    minimum_depth_m: float = 0.20,
    maximum_depth_m: float = 8.0,
    minimum_mask_pixels: int = 60,
    minimum_depth_coverage: float = 0.65,
    maximum_plane_residual_m: float = 0.0020,
    maximum_center_uncertainty_m: float = 0.00050,
    maximum_orientation_uncertainty_rad: float = 0.05,
    depth_noise_floor_m: float = 0.00020,
    maximum_fit_points: int = 3000,
) -> TargetTrackingResult:
    """Estimate the book face from one registered RGB-D pair.

    Normal visibility/depth failures return a failed result instead of raising.
    Invalid configuration still raises ``ValueError`` so it cannot be silently
    treated as a transient occlusion.
    """

    if not isinstance(rgb_image, np.ndarray) or not isinstance(
        depth_image, np.ndarray
    ):
        raise TypeError('rgb_image and depth_image must be NumPy arrays')
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError('rgb_image must be a BGR image')
    if depth_image.ndim == 3 and depth_image.shape[2] == 1:
        depth_image = depth_image[:, :, 0]
    if depth_image.ndim != 2:
        raise ValueError('depth_image must be single-channel')
    if rgb_image.shape[:2] != depth_image.shape:
        return _result(False, 'rgb_depth_shape_mismatch')
    if not 1 <= int(row) <= 4:
        raise ValueError('row must be from 1 through 4')
    if stamp_ns < 0:
        raise ValueError('stamp_ns cannot be negative')
    if depth_scale <= 0.0 or minimum_depth_m < 0.0:
        raise ValueError('depth scale and limits must be positive')
    if maximum_depth_m <= minimum_depth_m:
        raise ValueError('maximum_depth_m must exceed minimum_depth_m')
    if minimum_mask_pixels < 12 or maximum_fit_points < 12:
        raise ValueError('point-count limits are too small')
    if not 0.0 < minimum_depth_coverage <= 1.0:
        raise ValueError('minimum_depth_coverage must be in (0, 1]')
    if maximum_plane_residual_m <= 0.0 or maximum_center_uncertainty_m <= 0.0:
        raise ValueError('uncertainty limits must be positive')

    camera = _intrinsics(intrinsics)
    height, width = depth_image.shape
    x, y, box_width, box_height = map(int, detection.bbox)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + box_width), min(height, y + box_height)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return _result(False, 'target_bbox_outside_image')

    mask = hsv_color_mask(rgb_image, detection.color)
    roi_mask = mask[y1:y2, x1:x2] != 0
    rows, columns = np.nonzero(roi_mask)
    mask_count = len(rows)
    if mask_count < minimum_mask_pixels:
        return _result(
            False,
            'target_occluded',
            visible_mask_pixels=float(mask_count),
            minimum_mask_pixels=float(minimum_mask_pixels),
        )
    image_u = columns.astype(np.int64) + x1
    image_v = rows.astype(np.int64) + y1

    raw_depth = depth_image[image_v, image_u].astype(np.float64) * depth_scale
    valid = (
        np.isfinite(raw_depth)
        & (raw_depth >= minimum_depth_m)
        & (raw_depth <= maximum_depth_m)
    )
    valid_count = int(np.count_nonzero(valid))
    depth_coverage = valid_count / float(mask_count)
    if depth_coverage < minimum_depth_coverage or valid_count < 12:
        return _result(
            False,
            'insufficient_depth_coverage',
            depth_coverage=depth_coverage,
            valid_depth_pixels=float(valid_count),
            visible_mask_pixels=float(mask_count),
        )

    image_u = image_u[valid]
    image_v = image_v[valid]
    z = raw_depth[valid]
    points = np.column_stack(
        (
            (image_u - camera.cx) * z / camera.fx,
            (image_v - camera.cy) * z / camera.fy,
            z,
        )
    )
    try:
        plane_center, normal, residuals, plane_inliers = _dominant_front_plane(
            points,
            image_u,
            image_v,
            maximum_fit_points=maximum_fit_points,
        )
    except (ValueError, np.linalg.LinAlgError):
        return _result(False, 'inconsistent_depth_geometry')
    plane_residual = float(np.sqrt(np.mean(np.square(residuals))))
    if (
        not math.isfinite(plane_residual)
        or plane_residual > maximum_plane_residual_m
    ):
        return _result(
            False,
            'inconsistent_depth_geometry',
            plane_residual_m=plane_residual,
            maximum_plane_residual_m=maximum_plane_residual_m,
        )

    mask_points = np.column_stack(
        (
            image_u[plane_inliers].astype(np.float32),
            image_v[plane_inliers].astype(np.float32),
        )
    )
    rectangle = cv2.minAreaRect(mask_points.reshape(-1, 1, 2))
    pixel_extent = tuple(float(item) for item in rectangle[1])
    if min(pixel_extent) < 2.0:
        return _result(
            False,
            'target_occluded',
            minimum_pixel_extent=min(pixel_extent),
        )
    image_center = (float(rectangle[0][0]), float(rectangle[0][1]))
    image_corners = _ordered_quad(cv2.boxPoints(rectangle))
    try:
        center = _point_on_plane(image_center, plane_center, normal, camera)
        corners_array = np.asarray(
            [
                _point_on_plane(pixel, plane_center, normal, camera)
                for pixel in image_corners
            ]
        )
    except ValueError:
        return _result(False, 'inconsistent_depth_geometry')

    left_mid = (corners_array[0] + corners_array[3]) * 0.5
    right_mid = (corners_array[1] + corners_array[2]) * 0.5
    top_mid = (corners_array[0] + corners_array[1]) * 0.5
    bottom_mid = (corners_array[2] + corners_array[3]) * 0.5
    horizontal = right_mid - left_mid
    vertical = bottom_mid - top_mid
    horizontal_extent = float(np.linalg.norm(horizontal))
    vertical_extent = float(np.linalg.norm(vertical))
    if min(horizontal_extent, vertical_extent) <= 1e-5:
        return _result(False, 'inconsistent_face_extent')
    if vertical_extent >= horizontal_extent:
        long_axis = _normalize(vertical)
    else:
        long_axis = _normalize(horizontal)
    short_extent = min(horizontal_extent, vertical_extent)
    long_extent = max(horizontal_extent, vertical_extent)

    # The centre is inferred from both fitted face edges, so its image
    # quantisation is materially below one whole pixel.  Keep a non-zero floor
    # and report it instead of claiming exact sub-millimetre truth.
    minimum_pixel_extent = max(1.0, min(pixel_extent))
    center_pixel_uncertainty = max(
        0.06,
        0.50 / math.sqrt(minimum_pixel_extent),
    )
    lateral_uncertainty = (
        center_pixel_uncertainty
        * float(center[2])
        / min(camera.fx, camera.fy)
    )
    effective_samples = math.sqrt(max(1.0, min(float(len(residuals)), 1600.0)))
    residual_sigma = 1.4826 * float(
        np.median(np.abs(residuals - np.median(residuals)))
    )
    depth_uncertainty = (
        max(depth_noise_floor_m, residual_sigma) / effective_samples
    )
    center_uncertainty = math.hypot(lateral_uncertainty, depth_uncertainty)
    extent_uncertainty = 2.0 * center_uncertainty
    orientation_uncertainty = max(
        math.atan2(extent_uncertainty, max(long_extent, 1e-6)),
        math.atan2(
            max(depth_uncertainty, plane_residual / effective_samples),
            max(short_extent, 1e-6),
        ),
    )
    if center_uncertainty > maximum_center_uncertainty_m:
        return _result(
            False,
            'center_uncertainty_too_large',
            center_uncertainty_m=center_uncertainty,
            maximum_center_uncertainty_m=maximum_center_uncertainty_m,
        )
    if orientation_uncertainty > maximum_orientation_uncertainty_rad:
        return _result(
            False,
            'orientation_uncertainty_too_large',
            orientation_uncertainty_rad=orientation_uncertainty,
            maximum_orientation_uncertainty_rad=(
                maximum_orientation_uncertainty_rad
            ),
        )

    residual_score = math.exp(-plane_residual / maximum_plane_residual_m)
    uncertainty_score = math.exp(
        -center_uncertainty / maximum_center_uncertainty_m
    )
    quality = float(
        np.clip(
            float(detection.confidence)
            * min(1.0, depth_coverage / 0.90)
            * residual_score
            * uncertainty_score,
            0.0,
            1.0,
        )
    )
    observation = TargetBookObservation(
        stamp_ns=int(stamp_ns),
        frame_id=str(frame_id),
        target_colour=str(detection.color).lower(),
        row=int(row),
        center=tuple(float(item) for item in center),
        corners=tuple(
            tuple(float(item) for item in corner) for corner in corners_array
        ),  # type: ignore[arg-type]
        face_normal=tuple(float(item) for item in normal),
        long_axis=tuple(float(item) for item in long_axis),
        short_extent_m=short_extent,
        long_extent_m=long_extent,
        bbox=(x1, y1, x2 - x1, y2 - y1),
        image_center=image_center,
        detection_confidence=float(detection.confidence),
        depth_coverage=depth_coverage,
        plane_residual_m=plane_residual,
        center_uncertainty_m=center_uncertainty,
        extent_uncertainty_m=extent_uncertainty,
        orientation_uncertainty_rad=orientation_uncertainty,
        quality_confidence=quality,
    )
    return _result(
        True,
        'estimated',
        observation,
        depth_coverage=depth_coverage,
        plane_residual_m=plane_residual,
        center_uncertainty_m=center_uncertainty,
        orientation_uncertainty_rad=orientation_uncertainty,
    )


def _bbox_iou(first: BBox, second: BBox) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return float(intersection) / float(union) if union > 0 else 0.0


def _angle(first: Sequence[float], second: Sequence[float]) -> float:
    a, b = _normalize(np.asarray(first)), _normalize(np.asarray(second))
    return math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))


class TargetBookTracker:
    """Maintain target identity and fail-closed high-rate availability."""

    def __init__(
        self,
        *,
        minimum_rate_hz: float = 10.0,
        minimum_consecutive_observations: int = 3,
        maximum_age_seconds: float = 0.15,
        maximum_reacquisition_gap_seconds: float = 0.25,
        maximum_center_speed_mps: float = 0.45,
        maximum_image_speed_px_s: float = 650.0,
        minimum_interframe_extent_ratio: float = 0.78,
        maximum_interframe_extent_ratio: float = 1.28,
        maximum_interframe_orientation_rad: float = 0.35,
        maximum_center_uncertainty_m: float = 0.00050,
    ) -> None:
        if minimum_rate_hz <= 0.0 or maximum_age_seconds <= 0.0:
            raise ValueError('tracking rate and age limits must be positive')
        if minimum_consecutive_observations < 1:
            raise ValueError(
                'minimum_consecutive_observations must be positive'
            )
        if maximum_reacquisition_gap_seconds <= 0.0:
            raise ValueError(
                'maximum_reacquisition_gap_seconds must be positive'
            )
        if not 0.0 < minimum_interframe_extent_ratio <= 1.0:
            raise ValueError('minimum extent ratio must be in (0, 1]')
        if maximum_interframe_extent_ratio < 1.0:
            raise ValueError('maximum extent ratio must be at least 1')
        self.minimum_rate_hz = float(minimum_rate_hz)
        self.minimum_consecutive = int(minimum_consecutive_observations)
        self.maximum_age_seconds = float(maximum_age_seconds)
        self.maximum_reacquisition_gap_seconds = float(
            maximum_reacquisition_gap_seconds
        )
        self.maximum_center_speed_mps = float(maximum_center_speed_mps)
        self.maximum_image_speed_px_s = float(maximum_image_speed_px_s)
        self.minimum_extent_ratio = float(minimum_interframe_extent_ratio)
        self.maximum_extent_ratio = float(maximum_interframe_extent_ratio)
        self.maximum_orientation = float(maximum_interframe_orientation_rad)
        self.maximum_center_uncertainty = float(maximum_center_uncertainty_m)
        self._generation = 0
        self._latest: Optional[TargetBookObservation] = None
        self._blocked_reason = 'target_not_observed'
        self._blocked_stamp_ns = -1
        self._consecutive = 0
        self._identity_break_pending = False

    @property
    def last_good(self) -> Optional[TargetBookObservation]:
        """Return the last accepted sample for candidate association only.

        Safety consumers must call :meth:`latest`, which also checks current
        visibility, cadence, warm-up, age, and uncertainty.
        """

        return self._latest

    def reset(self) -> None:
        self._generation += 1
        self._latest = None
        self._blocked_reason = 'target_not_observed'
        self._blocked_stamp_ns = -1
        self._consecutive = 0
        self._identity_break_pending = False

    def mark_unavailable(
        self,
        stamp_ns: int,
        reason: str,
    ) -> TargetTrackingResult:
        """Invalidate the public latest state without reusing a prior pose."""

        stamp_ns = int(stamp_ns)
        if self._latest is not None:
            # Even a short blind interval breaks the proof that a future
            # detection is the same physical instance.  It may be associated
            # spatially for reacquisition, but receives a new track ID.
            self._identity_break_pending = True
        self._blocked_reason = str(reason)
        self._blocked_stamp_ns = max(stamp_ns, self._blocked_stamp_ns)
        self._consecutive = 0
        return _result(False, self._blocked_reason)

    def observe(
        self,
        observation: TargetBookObservation,
    ) -> TargetTrackingResult:
        """Accept an estimate only when temporally/physically continuous."""

        previous = self._latest
        if observation.center_uncertainty_m > self.maximum_center_uncertainty:
            self.mark_unavailable(
                observation.stamp_ns,
                'center_uncertainty_too_large',
            )
            return _result(
                False,
                'center_uncertainty_too_large',
                center_uncertainty_m=observation.center_uncertainty_m,
            )
        if previous is None:
            if self._generation == 0:
                self._generation = 1
            accepted = replace(
                observation,
                track_id=(
                    f'{observation.target_colour}:row{observation.row}:'
                    f'track{self._generation}'
                ),
                track_generation=self._generation,
                sequence=1,
                identity_continuity_confidence=1.0,
            )
            self._latest = accepted
            self._blocked_reason = ''
            self._blocked_stamp_ns = -1
            self._consecutive = 1
            self._identity_break_pending = False
            return _result(True, 'accepted', accepted)

        if observation.stamp_ns <= previous.stamp_ns:
            self.mark_unavailable(
                observation.stamp_ns,
                'out_of_order_observation',
            )
            return _result(False, 'out_of_order_observation')
        if (
            observation.target_colour != previous.target_colour
            or observation.row != previous.row
        ):
            self.mark_unavailable(
                observation.stamp_ns,
                'target_identity_changed',
            )
            return _result(False, 'target_identity_changed')
        if observation.frame_id != previous.frame_id:
            self.mark_unavailable(
                observation.stamp_ns,
                'tracking_frame_changed',
            )
            return _result(False, 'tracking_frame_changed')

        period = (observation.stamp_ns - previous.stamp_ns) / 1e9
        if period > self.maximum_reacquisition_gap_seconds:
            # A long blind interval cannot preserve the old identity.  Start a
            # new generation and force consumers holding an old reference to
            # reject it, even after this new track warms up.
            self._generation += 1
            accepted = replace(
                observation,
                track_id=(
                    f'{observation.target_colour}:row{observation.row}:'
                    f'track{self._generation}'
                ),
                track_generation=self._generation,
                sequence=1,
                interframe_period_s=period,
                observed_rate_hz=1.0 / period,
                identity_continuity_confidence=0.0,
            )
            self._latest = accepted
            self._blocked_reason = ''
            self._blocked_stamp_ns = -1
            self._consecutive = 1
            self._identity_break_pending = False
            return _result(True, 'new_track_after_gap', accepted)

        center_motion = float(
            np.linalg.norm(
                np.asarray(observation.center) - np.asarray(previous.center)
            )
        )
        image_motion = math.hypot(
            observation.image_center[0] - previous.image_center[0],
            observation.image_center[1] - previous.image_center[1],
        )
        short_ratio = observation.short_extent_m / previous.short_extent_m
        long_ratio = observation.long_extent_m / previous.long_extent_m
        normal_change = _angle(observation.face_normal, previous.face_normal)
        axis_change = _angle(observation.long_axis, previous.long_axis)
        orientation_change = max(normal_change, axis_change)
        overlap = _bbox_iou(observation.bbox, previous.bbox)
        center_motion_cap = (
            self.maximum_center_speed_mps * period
            + observation.center_uncertainty_m
            + previous.center_uncertainty_m
        )
        image_motion_cap = self.maximum_image_speed_px_s * period + 2.0
        metrics = {
            'interframe_period_s': period,
            'observed_rate_hz': 1.0 / period,
            'center_motion_m': center_motion,
            'center_motion_cap_m': center_motion_cap,
            'image_motion_px': image_motion,
            'image_motion_cap_px': image_motion_cap,
            'bbox_iou': overlap,
            'short_extent_ratio': short_ratio,
            'long_extent_ratio': long_ratio,
            'orientation_change_rad': orientation_change,
        }
        if (
            center_motion > center_motion_cap
            or image_motion > image_motion_cap
        ):
            self.mark_unavailable(
                observation.stamp_ns,
                'target_identity_discontinuity',
            )
            return _result(False, 'target_identity_discontinuity', **metrics)
        if overlap < 0.02 and center_motion > min(
            0.012,
            center_motion_cap * 0.60,
        ):
            self.mark_unavailable(
                observation.stamp_ns,
                'target_identity_discontinuity',
            )
            return _result(False, 'target_identity_discontinuity', **metrics)
        if not (
            self.minimum_extent_ratio
            <= short_ratio
            <= self.maximum_extent_ratio
            and self.minimum_extent_ratio
            <= long_ratio
            <= self.maximum_extent_ratio
        ):
            self.mark_unavailable(
                observation.stamp_ns,
                'target_occluded_or_extent_changed',
            )
            return _result(
                False,
                'target_occluded_or_extent_changed',
                **metrics,
            )
        if orientation_change > self.maximum_orientation:
            self.mark_unavailable(
                observation.stamp_ns,
                'target_orientation_discontinuity',
            )
            return _result(
                False,
                'target_orientation_discontinuity',
                **metrics,
            )

        motion_score = math.exp(-center_motion / max(center_motion_cap, 1e-9))
        extent_score = math.exp(
            -max(abs(math.log(short_ratio)), abs(math.log(long_ratio)))
        )
        orientation_score = math.exp(
            -orientation_change / max(self.maximum_orientation, 1e-9)
        )
        continuity = float(min(motion_score, extent_score, orientation_score))
        if self._identity_break_pending:
            self._generation += 1
            accepted = replace(
                observation,
                track_id=(
                    f'{observation.target_colour}:row{observation.row}:'
                    f'track{self._generation}'
                ),
                track_generation=self._generation,
                sequence=1,
                interframe_period_s=period,
                observed_rate_hz=1.0 / period,
                identity_continuity_confidence=0.0,
            )
            self._latest = accepted
            self._blocked_reason = ''
            self._blocked_stamp_ns = -1
            self._consecutive = 1
            self._identity_break_pending = False
            return _result(True, 'reacquired_new_track', accepted, **metrics)
        accepted = replace(
            observation,
            track_id=previous.track_id,
            track_generation=previous.track_generation,
            sequence=previous.sequence + 1,
            interframe_period_s=period,
            observed_rate_hz=1.0 / period,
            identity_continuity_confidence=continuity,
        )
        maximum_period = 1.05 / self.minimum_rate_hz
        if self._blocked_reason or period > maximum_period:
            self._consecutive = 1
        else:
            self._consecutive += 1
        self._latest = accepted
        self._blocked_reason = ''
        self._blocked_stamp_ns = -1
        return _result(True, 'accepted', accepted, **metrics)

    def latest(
        self,
        now_ns: int,
        *,
        maximum_age_seconds: Optional[float] = None,
    ) -> TargetTrackingResult:
        """Return the latest sample only when every live guard is valid."""

        if self._latest is None:
            return _result(
                False,
                self._blocked_reason or 'target_not_observed',
            )
        age = (int(now_ns) - self._latest.stamp_ns) / 1e9
        maximum_age = (
            self.maximum_age_seconds
            if maximum_age_seconds is None
            else float(maximum_age_seconds)
        )
        if age < -0.02:
            return _result(False, 'observation_from_future', age_seconds=age)
        if age > maximum_age:
            return _result(
                False,
                'stale_target_observation',
                age_seconds=age,
                maximum_age_seconds=maximum_age,
            )
        if self._blocked_reason:
            return _result(False, self._blocked_reason, age_seconds=age)
        if self._consecutive < self.minimum_consecutive:
            return _result(
                False,
                'target_track_warming_up',
                self._latest,
                consecutive_observations=float(self._consecutive),
                required_observations=float(self.minimum_consecutive),
            )
        if (
            self._latest.sequence > 1
            and self._latest.observed_rate_hz + 1e-9 < self.minimum_rate_hz
        ):
            return _result(
                False,
                'tracking_rate_below_minimum',
                self._latest,
                observed_rate_hz=self._latest.observed_rate_hz,
                minimum_rate_hz=self.minimum_rate_hz,
            )
        return _result(
            True,
            'tracking',
            self._latest,
            age_seconds=age,
            consecutive_observations=float(self._consecutive),
            observed_rate_hz=self._latest.observed_rate_hz,
        )


def guard_target_motion(
    reference: TargetBookObservation,
    current: TargetBookObservation,
    *,
    maximum_center_motion_m: float = 0.001,
    maximum_corner_motion_m: Optional[float] = None,
    maximum_orientation_change_rad: Optional[float] = None,
) -> TargetTrackingResult:
    """Fail closed unless metric motion (including uncertainty) is bounded.

    This is suitable for proving that the book stayed fixed during a guarded
    shelf manoeuvre.  A controller following an expected moving pose should
    compare each observation with its expected pose instead of using this
    zero-motion helper.
    """

    if maximum_center_motion_m <= 0.0:
        raise ValueError('maximum_center_motion_m must be positive')
    if maximum_corner_motion_m is None:
        maximum_corner_motion_m = maximum_center_motion_m
    if maximum_corner_motion_m <= 0.0:
        raise ValueError('maximum_corner_motion_m must be positive')
    if (
        maximum_orientation_change_rad is not None
        and maximum_orientation_change_rad <= 0
    ):
        raise ValueError('maximum_orientation_change_rad must be positive')
    if (
        not reference.track_id
        or reference.track_id != current.track_id
        or reference.target_colour != current.target_colour
        or reference.row != current.row
        or reference.frame_id != current.frame_id
    ):
        return _result(False, 'target_identity_not_continuous')
    if current.stamp_ns <= reference.stamp_ns:
        return _result(False, 'target_observation_not_newer')

    center_motion = float(
        np.linalg.norm(
            np.asarray(current.center) - np.asarray(reference.center)
        )
    )
    center_upper = (
        center_motion
        + reference.center_uncertainty_m
        + current.center_uncertainty_m
    )
    corner_motion = max(
        float(np.linalg.norm(np.asarray(after) - np.asarray(before)))
        for before, after in zip(reference.corners, current.corners)
    )
    corner_upper = (
        corner_motion
        + reference.extent_uncertainty_m
        + current.extent_uncertainty_m
    )
    orientation_change = max(
        _angle(reference.face_normal, current.face_normal),
        _angle(reference.long_axis, current.long_axis),
    )
    orientation_upper = (
        orientation_change
        + reference.orientation_uncertainty_rad
        + current.orientation_uncertainty_rad
    )
    metrics = {
        'center_motion_m': center_motion,
        'center_motion_upper_bound_m': center_upper,
        'maximum_center_motion_m': maximum_center_motion_m,
        'maximum_corner_motion_m': float(maximum_corner_motion_m),
        'corner_motion_m': corner_motion,
        'corner_motion_upper_bound_m': corner_upper,
        'orientation_change_rad': orientation_change,
        'orientation_change_upper_bound_rad': orientation_upper,
    }
    if center_upper > maximum_center_motion_m:
        return _result(
            False,
            'target_center_motion_exceeded',
            current,
            **metrics,
        )
    if corner_upper > maximum_corner_motion_m:
        return _result(
            False,
            'target_corner_motion_exceeded',
            current,
            **metrics,
        )
    if (
        maximum_orientation_change_rad is not None
        and orientation_upper > maximum_orientation_change_rad
    ):
        return _result(
            False,
            'target_orientation_motion_exceeded',
            current,
            **metrics,
        )
    return _result(True, 'target_motion_within_bounds', current, **metrics)


__all__ = [
    'TargetBookObservation',
    'TargetBookTracker',
    'TargetTrackingResult',
    'estimate_target_book_observation',
    'guard_target_motion',
]
