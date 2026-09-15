"""Local exact-array fast paths; no collision verdict or rounded transform."""
from collections import OrderedDict
from threading import Lock

import numpy as np


def any_boolean(mask):
    """Keep caller comparison/short-circuit order and custom NumPy dispatch."""
    if type(mask) is np.ndarray and mask.dtype == np.dtype(np.bool_):
        return mask.any()
    return np.any(mask)


class ExactRigidValidation:
    """64 exact 128-byte successful inputs, owned by one table instance.

    Only ordinary native float64, owned, C-contiguous 4x4 inputs participate.
    A fresh bytes snapshot supplies both the key and original validator input.
    Misses run the original validator; invalid inputs are never inserted. Hits
    return fresh immutable metadata. The original validator and its globals
    are fixed implementation dependencies, not caller-supplied policy updates.
    A replaced validator callable falls through on every call. No transform is
    rounded and no scene, margin, mesh, collision result or min-Z is retained.
    """
    MAXIMUM_ENTRIES = 64

    def __init__(self, validator):
        self._validator = validator
        self._entries = OrderedDict()
        self._lock = Lock()
        self.hits = self.misses = self.fallbacks = 0

    def validate(self, value, validator):
        if (validator is not self._validator or type(value) is not np.ndarray
                or value.dtype != np.dtype(np.float64) or value.shape != (4, 4)
                or not value.flags.c_contiguous or not value.flags.owndata
                or value.base is not None):
            self.fallbacks += 1
            return validator(value)
        content = value.tobytes(order='C')
        with self._lock:
            if content in self._entries:
                self._entries.move_to_end(content)
                self.hits += 1
                return np.frombuffer(content, dtype=np.float64).reshape(4, 4)
        self.misses += 1
        snapshot = np.frombuffer(content, dtype=np.float64).reshape(4, 4)
        checked = validator(snapshot)
        # Successful validation must leave every numerical input byte intact.
        if (type(checked) is not np.ndarray or checked.dtype != np.dtype(np.float64)
                or checked.shape != (4, 4) or checked.tobytes(order='C') != content):
            return checked
        with self._lock:
            if content not in self._entries:
                while len(self._entries) >= self.MAXIMUM_ENTRIES:
                    self._entries.popitem(last=False)
                self._entries[content] = None
            self._entries.move_to_end(content)
        return np.frombuffer(content, dtype=np.float64).reshape(4, 4)
