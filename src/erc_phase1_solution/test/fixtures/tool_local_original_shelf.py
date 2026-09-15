"""Exact R51 method; original full module SHA e414f86addb8b2c3931df81a440f57e60b8bc9354026bb0fda2156d3de6950bc."""
from __future__ import annotations

class ShelfCradleGeometry:
    def local_surfaces(
        self, aperture: float, finger_positions: Mapping[str, float] | None = None
    ) -> dict[str, np.ndarray]:
        """Resolve nominal mimic values, or use explicitly supplied joint values."""
        values = dict(finger_positions or {})
        values.setdefault('gripper_left_finger_joint', float(aperture))
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError('nonfinite gripper joint position')
        key = tuple(sorted(values.items()))
        if key in self._local_cache:
            return self._local_cache[key]

        def value(name: str) -> float:
            if name in values:
                return values[name]
            mimic = self.mimics.get(name)
            if mimic is None:
                return 0.0
            return (
                value(mimic.get('joint')) * float(mimic.get('multiplier', '1'))
                + float(mimic.get('offset', '0'))
            )

        transforms = {
            link: chain.forward([value(name) for name in chain.active_names])
            for link, chain in self.chains.items()
        }
        grasp = self.grasp_chain.forward(
            [value(name) for name in self.grasp_chain.active_names]
        )
        surfaces = {
            link: (triangles - grasp[:3, 3]) @ grasp[:3, :3]
            for link, triangles in world_gripper_surfaces(self.model, transforms).items()
        }
        self._local_cache[key] = surfaces
        return surfaces
