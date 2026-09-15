"""Bounded immutable local mesh snapshots and exact bounds, never verdicts."""
from collections import OrderedDict
from threading import Lock

import numpy as np


class ModelLocalMesh:
    """Explicit model-owned immutable triangles and exact extrema, not a verdict.

    Construction copies an ordinary owned array into bytes once. Every accessor
    returns fresh array metadata over immutable bytes, so reshaping a returned
    view cannot change a later snapshot. Raw/custom arrays never gain this
    contract merely by being read-only. Link, component order and watertight
    metadata stay in the owning CollisionMesh record.
    """

    __slots__ = ('_content', '_shape', '_low', '_high')
    MAXIMUM_ENTRIES = 64
    MAXIMUM_BYTES = 32*1024*1024

    def __init__(self, surface):
        if (type(surface) is not np.ndarray or surface.dtype != np.dtype(np.float64)
                or not surface.flags.c_contiguous or not surface.flags.owndata
                or surface.base is not None or surface.ndim != 3
                or surface.shape[1:] != (3, 3) or not len(surface)
                or surface.nbytes + 48 > self.MAXIMUM_BYTES):
            raise ValueError('model mesh requires a bounded owned float64 triangle array')
        content = surface.tobytes(order='C')
        snapshot = np.frombuffer(content, dtype=np.float64).reshape(surface.shape)
        if not np.all(np.isfinite(snapshot)):
            raise ValueError('model mesh requires finite triangles')
        low = snapshot.min(axis=(0, 1))
        high = snapshot.max(axis=(0, 1))
        object.__setattr__(self, '_content', content)
        object.__setattr__(self, '_shape', tuple(surface.shape))
        object.__setattr__(self, '_low', low.tobytes())
        object.__setattr__(self, '_high', high.tobytes())

    @staticmethod
    def from_immutable_surface(surface):
        """Rebuild a bounded handle over one complete immutable bytes payload.

        This is explicit ownership of already immutable geometry, not an
        identity shortcut for an arbitrary read-only or aliased mutable array.
        No geometry bytes are copied; extrema keep the original reduction order.
        """
        if (type(surface) is not np.ndarray or surface.dtype != np.dtype(np.float64)
                or not surface.flags.c_contiguous or surface.flags.writeable
                or surface.ndim != 3 or surface.shape[1:] != (3, 3)
                or not len(surface) or surface.nbytes + 48 > ModelLocalMesh.MAXIMUM_BYTES):
            raise ValueError('model mesh requires a complete immutable float64 triangle surface')
        owner = surface
        while type(owner) is np.ndarray:
            owner = owner.base
        if type(owner) is not bytes or len(owner) != surface.nbytes:
            raise ValueError('model mesh requires its complete immutable bytes owner')
        # Fresh metadata prevents caller changes to shape/dtype from affecting us.
        shape = tuple(surface.shape)
        snapshot = np.frombuffer(owner, dtype=np.float64).reshape(shape)
        if not np.all(np.isfinite(snapshot)):
            raise ValueError('model mesh requires finite triangles')
        low = snapshot.min(axis=(0, 1))
        high = snapshot.max(axis=(0, 1))
        result = object.__new__(ModelLocalMesh)
        object.__setattr__(result, '_content', owner)
        object.__setattr__(result, '_shape', shape)
        object.__setattr__(result, '_low', low.tobytes())
        object.__setattr__(result, '_high', high.tobytes())
        return result

    def __setattr__(self, name, value):
        raise AttributeError('model mesh snapshot is immutable')

    def __delattr__(self, name):
        raise AttributeError('model mesh snapshot is immutable')

    @property
    def nbytes(self):
        return len(self._content) + len(self._low) + len(self._high)

    def snapshot(self):
        """Fresh (triangles, minimum, maximum) views of one immutable snapshot."""
        return (np.frombuffer(self._content, dtype=np.float64).reshape(self._shape),
                np.frombuffer(self._low, dtype=np.float64),
                np.frombuffer(self._high, dtype=np.float64))

    def __array__(self, dtype=None, copy=None):
        surface = self.snapshot()[0]
        if copy is True:
            return np.array(surface, dtype=dtype, copy=True)
        return np.asarray(surface, dtype=dtype)

    def matches(self, surface):
        """Require the whole immutable buffer, exact shape/dtype and layout.

        This identity concerns immutable byte ownership, never mutable content,
        transforms, collision outcomes or raw arrays with writeable=False.
        """
        if (type(surface) is not np.ndarray or surface.dtype != np.dtype(np.float64)
                or surface.shape != self._shape or not surface.flags.c_contiguous
                or surface.flags.writeable or surface.nbytes != len(self._content)):
            return False
        owner = surface
        while type(owner) is np.ndarray:
            owner = owner.base
        return owner is self._content


class ExactLocalBounds:
    """Exact content-keyed snapshots; unsupported caller arrays fall through.

    Each caller must use the returned snapshot for both broad and narrow phase.
    Raw-array content bytes are read anew on every call; mutable object identity
    is never a key. Explicit model handles use already immutable bytes instead.
    The byte limit counts retained payload and bound bytes; the entry limit also
    bounds Python container overhead. Concurrent mutation during the byte copy
    is not an atomic sensor snapshot contract and is not supported by callers.
    """

    def __init__(self, *, maximum_entries=64, maximum_bytes=32*1024*1024):
        if (type(maximum_entries) is not int or maximum_entries <= 0
                or type(maximum_bytes) is not int or maximum_bytes <= 0):
            raise ValueError('local bounds capacities must be positive integers')
        self.maximum_entries = maximum_entries
        self.maximum_bytes = maximum_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = Lock()

    @staticmethod
    def model_snapshot(surface):
        return surface.snapshot() if type(surface) is ModelLocalMesh else None

    def capture(self, surface):
        """Return (immutable surface, immutable min, immutable max), or None."""
        if type(surface) is ModelLocalMesh:
            # The model owns this bounded payload; this cache retains no handle
            # or additional entry. No bytes/hash/equality scan is repeated.
            return surface.snapshot()
        if (type(surface) is not np.ndarray or surface.dtype != np.dtype(np.float64)
                or not surface.flags.c_contiguous or not surface.flags.owndata
                or surface.base is not None or surface.ndim != 3
                or surface.shape[1:] != (3, 3) or not len(surface)
                or surface.nbytes + 48 > self.maximum_bytes):
            return None
        content = surface.tobytes(order='C')
        key = (surface.shape, content)
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._entries.move_to_end(key)
                return existing
        snapshot = np.frombuffer(content, dtype=np.float64).reshape(surface.shape)
        if not np.all(np.isfinite(snapshot)):
            return None
        # Match the original arithmetic and its min-before-max order exactly.
        low = snapshot.min(axis=(0, 1))
        high = snapshot.max(axis=(0, 1))
        low = np.frombuffer(low.tobytes(), dtype=np.float64)
        high = np.frombuffer(high.tobytes(), dtype=np.float64)
        entry = (snapshot, low, high)
        weight = len(content) + low.nbytes + high.nbytes
        with self._lock:
            # Another caller may have populated the same immutable content.
            existing = self._entries.get(key)
            if existing is not None:
                self._entries.move_to_end(key)
                return existing
            while (self._entries and
                   (len(self._entries) >= self.maximum_entries
                    or self._bytes + weight > self.maximum_bytes)):
                old_key, _ = self._entries.popitem(last=False)
                self._bytes -= len(old_key[1]) + 48
            self._entries[key] = entry
            self._bytes += weight
        return entry
