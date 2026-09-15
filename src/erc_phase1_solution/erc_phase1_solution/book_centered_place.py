"""Construct a nominal book-centred target from onboard point and held box.

The legacy point is a visual interior reference. Registered placement instead
supplies a fitted cavity-floor center and CAD orientation. This helper supplies
a target only; existing motion/collision checks still apply.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BookCenteredPlaceTarget:
    book_center: np.ndarray
    tool_position: np.ndarray
    tool_rotation: np.ndarray
    attachment_center: np.ndarray
    book_axes_in_tool: np.ndarray


def book_centered_place_target(corners_in_tool, observed_point, height_above_point,
                              *, bin_rotation=None):
    """Align the book to a measured CAD bin frame, or the legacy approach ray.

    Corners use the existing x/y/z sign-product order. Their existing padding
    changes edge lengths but neither axes nor centre; it is not added again.
    A supplied bin frame has CAD X across, Y up and Z along the cavity. Its
    point must be the measured cavity-floor center. Pose/freshness admission
    belongs to the caller; this helper validates the rigid upright frame.
    """
    corners = np.asarray(corners_in_tool, dtype=float)
    point = np.asarray(observed_point, dtype=float)
    height = float(height_above_point)
    if corners.shape != (8, 3) or not np.all(np.isfinite(corners)):
        raise ValueError('held book requires eight finite ordered corners')
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError('bin reference requires a finite three-dimensional point')
    if not np.isfinite(height) or height <= 0.0:
        raise ValueError('book centre height above reference must be positive')
    edges = np.column_stack((corners[4]-corners[0], corners[2]-corners[0], corners[1]-corners[0]))
    lengths = np.linalg.norm(edges, axis=0)
    if np.any(lengths <= 1e-9):
        raise ValueError('held book has a degenerate axis')
    axes = edges / lengths
    if not np.allclose(axes.T @ axes, np.eye(3), rtol=0.0, atol=1e-7) or np.linalg.det(axes) <= 0.0:
        raise ValueError('held book axes must form a right-handed orthonormal frame')
    center = np.mean(corners, axis=0)
    signs = np.asarray([(x,y,z) for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
    reconstructed = center + (signs * (lengths / 2.0)) @ axes.T
    if not np.allclose(corners, reconstructed, rtol=0.0, atol=1e-7):
        raise ValueError('held book corner ordering is inconsistent')
    if bin_rotation is None:
        horizontal_distance = float(np.linalg.norm(point[:2]))
        if horizontal_distance <= 1e-9:
            raise ValueError('bin reference has no horizontal approach direction')
        along = np.asarray([point[0], point[1], 0.0]) / horizontal_distance
        left = np.asarray([-along[1], along[0], 0.0])
        up = np.asarray([0., 0., 1.])
    else:
        frame = np.asarray(bin_rotation, dtype=float)
        if (frame.shape != (3, 3) or not np.all(np.isfinite(frame))
                or not np.allclose(frame.T @ frame, np.eye(3), rtol=0., atol=1e-7)
                or not np.isclose(np.linalg.det(frame), 1., rtol=0., atol=1e-7)
                or frame[2, 1] < np.cos(np.deg2rad(5.))):
            raise ValueError('bin rotation must be a rigid upright CAD frame')
        left, up, along = frame[:, 0], frame[:, 1], frame[:, 2]
    desired_axes = np.column_stack((-left, -up, along))
    rotation = desired_axes @ axes.T
    desired_center = point + up * height
    return BookCenteredPlaceTarget(
        book_center=desired_center,
        tool_position=desired_center - rotation @ center,
        tool_rotation=rotation,
        attachment_center=center,
        book_axes_in_tool=axes,
    )
