"""RGB-D bin geometry and freshness checks, independent of simulator truth.

Adapted from CLIO ERC's MIT-licensed bin_perception.py; see
THIRD_PARTY_NOTICES.md for source revision and license.
"""
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .vision import BinDetection, hsv_color_mask


BIN_MAX_AGE_NS = 500_000_000
BIN_MAX_SKEW_NS = 50_000_000


def bin_stamp_is_fresh(stamp_ns: int, now_ns: int) -> bool:
    """Reject stale and future points, including latched points on reconnect."""
    return 0 <= now_ns - stamp_ns <= BIN_MAX_AGE_NS


def bin_message_is_current(message, now_ns: int, invalidated_ns: int = -1) -> bool:
    """Check freshness, validity and invalidation ordering for a bin point."""
    if message is None:
        return False
    stamp = message.header.stamp
    stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    point = message.point
    return bool(
        message.header.frame_id and stamp_ns > invalidated_ns
        and bin_stamp_is_fresh(stamp_ns, now_ns)
        and np.all(np.isfinite([point.x, point.y, point.z])) and point.z > 0.0
    )


def bin_message_is_verified(message, now_ns, verified_ns, invalidated_ns=-1):
    """A point and positive geometry status must refer to the same frame."""
    if not bin_message_is_current(message, now_ns, invalidated_ns):
        return False
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec) == verified_ns


def depth_in_metres(depth: np.ndarray) -> np.ndarray:
    """Convert ROS uint16 millimetres; floating depth is already in metres."""
    scale = 0.001 if depth.dtype.kind in 'ui' else 1.0
    return np.asarray(depth, dtype=np.float64) * scale


def red_bin_candidates(bgr, minimum_area=360.0):
    """Return every plausible red region so a large logo cannot hide a bin."""
    mask = hsv_color_mask(bgr, 'red', open_kernel=3, close_kernel=5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, width, height = cv2.boundingRect(contour)
        fill = area / max(1, width * height)
        if area >= minimum_area and min(width, height) >= 4 and fill >= 0.32:
            candidates.append(BinDetection(
                (x, y, width, height), area,
                min(1.0, 0.55 * fill + 0.45 * min(1.0, area / 1800.0)),
            ))
    return sorted(candidates, key=lambda candidate: candidate.area, reverse=True)


def bin_surface_camera_point(bgr, depth, intrinsics, detection):
    """Keep the placement contract: front surface at 70% of the image box.

    Only red pixels with valid depth enter the patch median; background table
    pixels must not move the target behind the observed bin wall.
    """
    x, y, width, height = detection.bbox
    u, v = x + width / 2.0, y + height * 0.70
    px, py = int(round(u)), int(round(v))
    depth = depth_in_metres(np.asarray(depth))
    xa, xb = max(0, px-6), min(depth.shape[1], px+7)
    ya, yb = max(0, py-6), min(depth.shape[0], py+7)
    patch = depth[ya:yb, xa:xb]
    red = hsv_color_mask(bgr, 'red')[ya:yb, xa:xb] > 0
    valid = red & np.isfinite(patch) & (patch > 0.20) & (patch < 8.0)
    if np.count_nonzero(valid) < 3:
        return None
    z = float(np.median(patch[valid]))
    k = np.asarray(intrinsics).reshape(-1)
    return np.array([(u-k[2])*z/k[0], (v-k[5])*z/k[4], z])


@dataclass
class BinObservation:
    detection: BinDetection
    camera_point: np.ndarray
    base_point: np.ndarray
    metrics: dict = field(default_factory=dict)


def camera_transform(translation, quaternion) -> np.ndarray:
    """Homogeneous camera-to-base transform from a ROS Transform message."""
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    norm = x*x + y*y + z*z + w*w
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError('Invalid camera transform')
    s = 2.0 / norm
    result = np.eye(4)
    result[:3, :3] = [
        [1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
        [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
        [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)]]
    result[:3, 3] = [translation.x, translation.y, translation.z]
    return result


def verify_bin_candidate(bgr, depth, intrinsics, base_from_camera, detection,
                         table_height=0.73):
    """Require bin-sized red geometry supported by a visible horizontal table.

    Dimensions come from the task specification (0.50 x 0.31 x 0.21 m bin,
    0.73 m table), not the world's bin location. Missing geometry fails closed.
    Returns (observation or None, diagnostic reason).
    """
    if depth is None or depth.shape != bgr.shape[:2]:
        return None, 'unaligned_depth'
    depth = depth_in_metres(np.asarray(depth))
    k = np.asarray(intrinsics).reshape(-1)
    transform = np.asarray(base_from_camera)
    if (k.size != 9 or not np.all(np.isfinite(k)) or min(k[0], k[4]) <= 0 or
            transform.shape != (4, 4) or not np.all(np.isfinite(transform))):
        return None, 'invalid_calibration'
    x, y, w, h = detection.bbox
    height, width = depth.shape
    if x < 2 or y < 2 or x+w >= width-2 or y+h >= height-2:
        return None, 'clipped_candidate'
    # Inspect the object and nearby support surface at half image resolution.
    xa, xb = max(0, x-w), min(width, x+2*w)
    ya, yb = max(0, y-h//2), min(height, y+3*h)
    v, u = np.mgrid[ya:yb:2, xa:xb:2]
    z = depth[ya:yb:2, xa:xb:2].astype(np.float64)
    valid = np.isfinite(z) & (z > 0.15) & (z < 8.0)
    camera = np.stack(((u-k[2])*z/k[0], (v-k[5])*z/k[4], z), axis=-1)
    # Invalid depths must not pollute normal/plane computations downstream.
    camera[~valid] = 0
    points = np.einsum('ij,hwj->hwi', transform[:3, :3], camera) + transform[:3, 3]
    red = hsv_color_mask(bgr[ya:yb, xa:xb], 'red')[::2, ::2] > 0
    inside = (u >= x) & (u < x+w) & (v >= y) & (v < y+h)
    object_mask = valid & red & inside
    if np.count_nonzero(object_mask) < 25:
        return None, 'insufficient_object_depth'
    object_points = points[object_mask]
    camera_points = camera[object_mask]
    target = np.median(camera_points, axis=0)
    if target[2] < 0.58:
        return None, 'inside_held_book_range'
    low, high = np.quantile(object_points[:, 2], [0.05, 0.95])
    if low < table_height-0.07 or high > table_height+0.29:
        return None, 'wrong_height_above_floor'
    extent_z = float(high-low)
    if not 0.10 <= extent_z <= 0.27:
        return None, 'not_upright_bin_geometry'
    centered = object_points[:, :2] - np.median(object_points[:, :2], axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes[0]
    horizontal_extent = float(np.quantile(projected, 0.975)-np.quantile(projected, 0.025))
    if not 0.23 <= horizontal_extent <= 0.65:
        return None, 'wrong_physical_width'

    # A printed logo has almost the same 3-D plane as its non-red surroundings.
    center = np.median(object_points, axis=0)
    _, _, axes3 = np.linalg.svd(object_points-center, full_matrices=False)
    normal = axes3[-1]
    residuals = np.abs((object_points-center) @ normal)
    ring = (cv2.dilate(object_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    ring &= valid & ~red
    if np.count_nonzero(ring) >= 20 and np.quantile(residuals, 0.9) < 0.015:
        coplanar = np.mean(np.abs((points[ring]-center) @ normal) < 0.015)
        if coplanar > 0.80:
            return None, 'printed_on_surrounding_surface'

    # A wall can contain points at table height, but their surface normals are
    # vertical, not horizontal. Require a nearby, non-red horizontal patch.
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    normals = np.cross(dx, dy)
    lengths = np.linalg.norm(normals, axis=-1)
    horizontal = np.abs(normals[..., 2]) / np.maximum(lengths, 1e-12) > 0.90
    neighbors_valid = (valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] &
                       valid[2:, 1:-1] & valid[:-2, 1:-1])
    support_points = points[1:-1, 1:-1]
    support = (neighbors_valid & horizontal & (lengths > 1e-10) &
               # At 3--4 m range a grazing tabletop can span >10 cm across
               # four image pixels; retain it while rejecting large jumps.
               (np.linalg.norm(dx, axis=-1) < 0.30) &
               (np.linalg.norm(dy, axis=-1) < 0.30) &
               ~red[1:-1, 1:-1] &
               (np.abs(support_points[..., 2]-table_height) < 0.05) &
               (np.linalg.norm(support_points[..., :2]-center[:2], axis=-1) < 0.65))
    support_count = int(np.count_nonzero(support))
    if support_count < 30:
        return None, 'no_horizontal_table_support'
    return BinObservation(detection, target, center, {
        'width_m': round(horizontal_extent, 3), 'height_m': round(extent_z, 3),
        'support_pixels': support_count}), 'verified_geometry'


class BinTracker:
    """Only confirm distinct, recent frames observing one consistent object."""
    def __init__(self, required_frames=3):
        self.required_frames = max(3, required_frames)
        self.reset()

    def reset(self):
        self.last_stamp = None
        self.last_point = None
        self.count = 0

    def observe(self, observation: Optional[BinObservation], stamp_ns: int) -> bool:
        if observation is None:
            self.reset()
            return False
        if self.last_stamp is not None and stamp_ns <= self.last_stamp:
            return False
        continuous = (self.last_stamp is not None and
                      stamp_ns-self.last_stamp <= 400_000_000 and
                      np.linalg.norm(observation.base_point-self.last_point) <= 0.18)
        self.count = self.count+1 if continuous else 1
        self.last_stamp, self.last_point = stamp_ns, observation.base_point.copy()
        return self.count >= self.required_frames
