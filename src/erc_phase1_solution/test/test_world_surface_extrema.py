"""Exact bytes against original extrema, including exceptional/tie inputs."""
import numpy as np
import pytest
from erc_phase1_solution.exact_world_geometry import _surface_extrema, _WorldSurface


def original(surface):
    return np.asarray([np.min(surface, axis=(0, 1)), np.max(surface, axis=(0, 1))])


def immutable(surface):
    return np.frombuffer(surface.tobytes(), dtype=surface.dtype).reshape(surface.shape)


@pytest.mark.parametrize('count', [1, 12, 63, 64, 65, 513, 2048])
@pytest.mark.parametrize('kind', ['ordinary', 'all_equal', 'positive', 'negative',
    'subnormal', 'signed_zero', 'zero_low', 'zero_high', 'nan', 'inf'])
def test_bounds_and_frozen_bytes_match_original(count, kind):
    rng = np.random.default_rng(1973 + count)
    values = rng.normal(size=(count, 3, 3))
    if kind == 'all_equal': values[:] = .125
    elif kind == 'positive': values = np.abs(values) + .01
    elif kind == 'negative': values = -np.abs(values) - .01
    elif kind == 'subnormal': values *= np.nextafter(0., 1.)
    elif kind == 'signed_zero':
        values[:] = 0.; values.ravel()[::2] = -0.
    elif kind == 'zero_low': values = np.abs(values); values[0, 0, :] = [-0., 0., -0.]
    elif kind == 'zero_high': values = -np.abs(values); values[0, 0, :] = [0., -0., 0.]
    elif kind == 'nan': values[count//2, 1, 1] = np.nan
    elif kind == 'inf': values[count//2, 1, 1] = np.inf
    surface = immutable(values)
    expected = original(surface)
    assert _surface_extrema(surface).tobytes() == expected.tobytes()
    if np.isfinite(expected).all():
        world, bounds = _WorldSurface(values).views()
        assert world.tobytes() == values.tobytes()
        assert bounds.tobytes() == expected.tobytes()
    else:
        with pytest.raises(ValueError, match='world geometry is nonfinite'):
            _WorldSurface(values)


@pytest.mark.parametrize('kind', ['writeable', 'float32', 'fortran', 'strided', 'subclass'])
def test_unsupported_arrays_preserve_original_values_and_dispatch(kind):
    class Custom(np.ndarray):
        calls = []
        def __array_function__(self, func, types, args, kwargs):
            self.calls.append(func.__name__)
            return super().__array_function__(func, types, args, kwargs)
    surface = np.arange(1800., dtype=float).reshape(200, 3, 3).copy() + .25
    if kind == 'float32': surface = surface.astype(np.float32)
    elif kind == 'fortran': surface = np.asfortranarray(surface)
    elif kind == 'strided': surface = surface[::2]
    elif kind == 'subclass': surface = surface.view(Custom)
    expected = original(surface)
    if kind == 'subclass': surface.calls.clear()
    assert _surface_extrema(surface).tobytes() == expected.tobytes()
    if kind == 'subclass': assert surface.calls == ['amin', 'amax']


def test_empty_surface_keeps_original_rejection():
    with pytest.raises(ValueError, match='zero-size array'):
        _surface_extrema(np.empty((0, 3, 3)))
