"""Associate live book detections with the visually selected shelf bay.

The static shelf registration comes from onboard marker RGB-D and TF.  The
0.20 m lateral allowance is the same engineering allowance used by
``shelf_bay_context`` (150 mm glyph/card offset and 50 mm registration error).
With the official 250 mm book jitter, the selected bay occupies at most
450 mm about its observed marker; the nearest adjacent book is at least
550 mm away.  These bounds concern identity, not grasp or collision accuracy.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np


REGISTRATION_MAX_AGE_NS = 120_000_000_000
COMMAND_MAX_AGE_NS = 2_000_000_000
FRAME_MAX_AGE_NS = 200_000_000
LATERAL_LIMIT_M = 0.45
SHELF_DEPTH_LIMIT_M = 0.35
ROW_HEIGHT_LIMIT_M = 0.12  # Below half the official 0.33 m active-row spacing.


def _stamp(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'book_selection_invalid_{name}')
    return value


def _vector(value, size, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f'book_selection_invalid_{name}') from error
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'book_selection_invalid_{name}')
    return tuple(float(v) for v in result)


@dataclass(frozen=True)
class BookSelectionContext:
    column: int
    colour: str
    marker_odom: tuple
    outward_normal_odom: tuple
    observation_stamp_ns: int
    dispatch_stamp_ns: int
    confirmed_row: int | None = None
    confirmed_point_odom: tuple | None = None
    confirmed_point_stamp_ns: int | None = None

    def fields(self):
        return dict(schema=1, source='onboard_marker_cloud', frame_id='odom',
                    column=self.column, colour=self.colour,
                    marker_odom=list(self.marker_odom),
                    outward_normal_odom=list(self.outward_normal_odom),
                    observation_stamp_ns=self.observation_stamp_ns,
                    dispatch_stamp_ns=self.dispatch_stamp_ns,
                    confirmed_row=self.confirmed_row,
                    confirmed_point_odom=(None if self.confirmed_point_odom is None
                                          else list(self.confirmed_point_odom)),
                    confirmed_point_stamp_ns=self.confirmed_point_stamp_ns)

    def require_frame(self, rgb_ns, depth_ns, now_ns):
        _stamp(now_ns, 'clock')
        if not 0 <= now_ns - self.observation_stamp_ns <= REGISTRATION_MAX_AGE_NS:
            raise ValueError('book_selection_registration_stale')
        for value in (rgb_ns, depth_ns):
            _stamp(value, 'frame_stamp')
            if value < self.dispatch_stamp_ns or not 0 <= now_ns-value <= FRAME_MAX_AGE_NS:
                raise ValueError('book_selection_frame_outside_epoch')

    def contains(self, point_odom):
        point = np.asarray(_vector(point_odom, 3, 'point'))
        normal = np.asarray(self.outward_normal_odom)
        tangent = np.asarray([normal[1], -normal[0]])
        delta = point[:2] - self.marker_odom[:2]
        if (abs(float(delta @ tangent)) > LATERAL_LIMIT_M
                or abs(float(delta @ normal)) > SHELF_DEPTH_LIMIT_M):
            return False
        return (self.confirmed_point_odom is None
                or abs(point[2]-self.confirmed_point_odom[2]) <= ROW_HEIGHT_LIMIT_M)


def decode_context(raw, now_ns, *, column, colour, confirmed_row):
    """Validate a new mode command; a missing context never selects a neighbor."""
    if (not isinstance(raw, Mapping) or raw.get('schema') != 1
            or raw.get('source') != 'onboard_marker_cloud'
            or raw.get('frame_id') != 'odom'):
        raise ValueError('book_selection_context_missing_or_incompatible')
    selected_column = raw.get('column')
    if (isinstance(selected_column, bool) or not isinstance(selected_column, int)
            or not 1 <= selected_column <= 5 or selected_column != column
            or raw.get('colour') not in ('red', 'blue', 'green', 'yellow')
            or raw.get('colour') != colour):
        raise ValueError('book_selection_target_mismatch')
    marker = _vector(raw.get('marker_odom'), 3, 'marker')
    normal = _vector(raw.get('outward_normal_odom'), 2, 'normal')
    if not math.isclose(math.hypot(*normal), 1., abs_tol=1e-4):
        raise ValueError('book_selection_normal_not_unit')
    observed = _stamp(raw.get('observation_stamp_ns'), 'observation_stamp')
    dispatched = _stamp(raw.get('dispatch_stamp_ns'), 'dispatch_stamp')
    _stamp(now_ns, 'clock')
    if not 0 <= now_ns-observed <= REGISTRATION_MAX_AGE_NS:
        raise ValueError('book_selection_registration_stale')
    if not observed <= dispatched <= now_ns or now_ns-dispatched > COMMAND_MAX_AGE_NS:
        raise ValueError('book_selection_command_stale')
    row = raw.get('confirmed_row')
    point = raw.get('confirmed_point_odom')
    point_stamp = raw.get('confirmed_point_stamp_ns')
    if row is None:
        if confirmed_row is not None or point is not None or point_stamp is not None:
            raise ValueError('book_selection_confirmed_row_missing')
    else:
        if (isinstance(row, bool) or not isinstance(row, int) or not 1 <= row <= 4
                or row != confirmed_row):
            raise ValueError('book_selection_confirmed_row_mismatch')
        point = _vector(point, 3, 'confirmed_point')
        point_stamp = _stamp(point_stamp, 'confirmed_point_stamp')
        if not observed <= point_stamp <= dispatched:
            raise ValueError('book_selection_confirmed_point_epoch')
    result = BookSelectionContext(selected_column, colour, marker, normal,
                                  observed, dispatched, row, point, point_stamp)
    if point is not None and not result.contains(point):
        raise ValueError('book_selection_confirmed_point_outside_bay')
    return result


def make_context(marker, normal, observed_ns, dispatch_ns, column, colour,
                 *, confirmed_row=None, confirmed_point=None, confirmed_stamp_ns=None):
    raw = dict(schema=1, source='onboard_marker_cloud', frame_id='odom',
               column=column, colour=colour, marker_odom=marker,
               outward_normal_odom=normal, observation_stamp_ns=observed_ns,
               dispatch_stamp_ns=dispatch_ns, confirmed_row=confirmed_row,
               confirmed_point_odom=confirmed_point,
               confirmed_point_stamp_ns=confirmed_stamp_ns)
    return decode_context(raw, dispatch_ns, column=column, colour=colour,
                          confirmed_row=confirmed_row).fields()


def select_registered_book(books, context, deproject, odom_from_camera):
    """Return the unique target-color component inside the registered bay/row."""
    matrix = np.asarray(odom_from_camera, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError('book_selection_invalid_camera_transform')
    accepted = []
    for book in books:
        if book.color != context.colour:
            continue
        point = deproject(book.center)
        if point is None:
            continue
        try:
            camera_point = np.asarray(_vector(point, 3, 'camera_point'))
            world = matrix[:3, :3] @ camera_point + matrix[:3, 3]
            if context.contains(world):
                accepted.append((book, tuple(camera_point)))
        except ValueError:
            continue
    if len(accepted) != 1:
        reason = 'ambiguous_candidates' if accepted else 'target_outside_registered_bay'
        raise ValueError('book_selection_' + reason)
    return accepted[0]
