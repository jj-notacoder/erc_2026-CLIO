"""Sensor-frame transfer, timing, and fail-closed lift registration checks."""
import copy
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from erc_phase1_solution.shelf_bay_context import make_bay_context, decode_bay_context


NOW = 50_000_000_000


def context():
    return make_bay_context([3., 1., 2.26], [-1., 0.], 2, NOW-20_000_000_000, NOW)


def node(pose=(1., .5, 0.)):
    return SimpleNamespace(
        _lock=threading.RLock(),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=NOW)),
        _staging_odom=dict(stamp_ns=NOW, pose=pose, linear_speed=0., angular_speed=0.),
    )


@pytest.mark.parametrize('yaw', [0., math.pi/2, -math.pi/2, math.pi])
def test_visual_odom_point_and_inward_axis_use_current_base_transform(yaw):
    payload = {'shelf_bay_context': context()}
    before = copy.deepcopy(payload)
    bay = decode_bay_context(node((1., .5, yaw)), payload)
    c, s = math.cos(yaw), math.sin(yaw)
    assert bay.marker_center_base == pytest.approx([2*c+.5*s, -2*s+.5*c, 2.26])
    assert bay.inward_axis_base == pytest.approx([c, -s, 0.])
    assert bay.physical_column == 2
    assert bay.lateral_uncertainty_m == .2
    assert bay.roof_uncertainty_m == .01
    assert bay.assume_upright_supported
    assert 'covariance unmeasured' in bay.source
    assert payload == before


@pytest.mark.parametrize(('key', 'value'), [
    ('schema', 2), ('source', 'evaluator'), ('frame_id', 'world'),
    ('marker_odom', [1., 2.]), ('marker_odom', [1., np.nan, 3.]),
    ('outward_normal_odom', [2., 0.]), ('outward_normal_odom', [0., 0.]),
    ('physical_column', 0), ('physical_column', 6), ('physical_column', True),
    ('observation_stamp_ns', 0), ('observation_stamp_ns', NOW+1),
    ('observation_stamp_ns', -1), ('dispatch_stamp_ns', NOW+1),
    ('dispatch_stamp_ns', NOW-2_000_000_001),
    ('dispatch_stamp_ns', NOW-30_000_000_000),
])
def test_registration_rejects_wrong_frame_identity_geometry_and_epoch(key, value):
    invalid = context()
    invalid[key] = value
    with pytest.raises(ValueError):
        decode_bay_context(node(), {'shelf_bay_context': invalid})


@pytest.mark.parametrize('payload', [None, {}, {'shelf_bay_context': {}}, []])
def test_missing_registration_cannot_enable_lift(payload):
    with pytest.raises(ValueError):
        decode_bay_context(node(), payload)


@pytest.mark.parametrize(('key', 'value'), [
    ('stamp_ns', NOW-350_000_001), ('stamp_ns', NOW+50_000_001),
    ('pose', (1., 2., np.nan)), ('linear_speed', .0051),
    ('angular_speed', .0081), ('linear_speed', np.nan), ('angular_speed', -.1),
])
def test_registration_rejects_stale_invalid_or_moving_current_base(key, value):
    current = node()
    current._staging_odom[key] = value
    with pytest.raises(ValueError):
        decode_bay_context(current, {'shelf_bay_context': context()})


def test_registration_age_is_bounded_without_replacing_observation_stamp():
    with pytest.raises(ValueError, match='stale'):
        make_bay_context([3., 1., 2.26], [-1., 0.], 2, 1, 121_000_000_000)
    actual = context()
    assert actual['observation_stamp_ns'] == NOW-20_000_000_000
    assert actual['dispatch_stamp_ns'] == NOW
