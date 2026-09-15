"""Bounded handle registry for exact immutable local tool surfaces.

The owner's existing aperture geometry cache retains its original policy.
Its immutable entries replace their prior raw geometry payload, rather than
keeping a second full copy. Only this additional handle/bounds registry evicts.
"""
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
import numpy as np
from .exact_local_bounds import ModelLocalMesh


@dataclass(frozen=True)
class _FrozenToolSurface:
    content: bytes
    shape: tuple

    def surface(self):
        return np.frombuffer(self.content, dtype=np.float64).reshape(self.shape)


def _owner(surface):
    if type(surface) is not np.ndarray:
        return None
    value = surface
    while type(value) is np.ndarray:
        value = value.base
    return value if type(value) is bytes else None


class ToolLocalMeshes:
    def __init__(self, *, maximum_entries=64, maximum_bytes=32*1024*1024):
        if (type(maximum_entries) is not int or maximum_entries <= 0
                or type(maximum_bytes) is not int or maximum_bytes <= 0):
            raise ValueError('tool handle capacities must be positive integers')
        self.maximum_entries, self.maximum_bytes = maximum_entries, maximum_bytes
        self._entries, self._bytes = OrderedDict(), 0
        self._lock = Lock()

    def model_for_surface(self, surface):
        owner = _owner(surface)
        if owner is None:
            return None
        with self._lock:
            model = self._entries.get(id(owner))
            if model is None or not model.matches(surface):
                return None
            self._entries.move_to_end(id(owner))
            return model

    def _remember(self, model):
        if type(model) is not ModelLocalMesh or model.nbytes > self.maximum_bytes:
            return
        surface = model.snapshot()[0]
        owner = _owner(surface)
        with self._lock:
            existing = self._entries.get(id(owner))
            if existing is not None and existing.matches(surface):
                self._entries.move_to_end(id(owner))
                return
            while self._entries and (len(self._entries) >= self.maximum_entries
                    or self._bytes + model.nbytes > self.maximum_bytes):
                _, removed = self._entries.popitem(last=False)
                self._bytes -= removed.nbytes
            self._entries[id(owner)] = model
            self._bytes += model.nbytes

    def freeze(self, surfaces):
        """Called only on fresh built-in producer output, before it escapes."""
        result = {}
        for link, surface in surfaces.items():
            if (type(surface) is not np.ndarray
                    or surface.nbytes + 48 > self.maximum_bytes):
                result[link] = surface
                continue
            try:
                model = ModelLocalMesh(surface)
            except (TypeError, ValueError, OverflowError):
                # Preserve the original raw validation/error path and ordering.
                result[link] = surface
                continue
            self._remember(model)
            snapshot = model.snapshot()[0]
            result[link] = _FrozenToolSurface(_owner(snapshot), tuple(snapshot.shape))
        return result

    def materialize(self, cached):
        """Fresh views/dict; registry eviction never recomputes aperture FK."""
        result = {}
        for link, stored in cached.items():
            if type(stored) is not _FrozenToolSurface:
                result[link] = stored
                continue
            surface = stored.surface()
            if self.model_for_surface(surface) is None:
                try:
                    model = ModelLocalMesh.from_immutable_surface(surface)
                except (TypeError, ValueError, OverflowError):
                    model = None
                if model is not None:
                    self._remember(model)
            result[link] = surface
        return result
