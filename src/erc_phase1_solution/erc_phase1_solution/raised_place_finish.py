"""Select a checked raised-arm HOME finish only after measured normal release."""
from __future__ import annotations

from contextlib import nullcontext
import numpy as np

from .place_contact_guard import PlaceContactGuard
from .release_evidence import AttemptIdentity


def checked_enabled(value):
    if type(value) is not bool:
        raise ValueError('place_finish_keep_torso_height_enabled must be boolean')
    return value


def normal_finish(node, correlation, torso_ready, direct_empty_home, home):
    """Return explicit call options and the actual final measurement target.

    The caller owns normal PLACE ordering: its measured open has succeeded.
    The original scene plan includes HOME at this torso height and then the
    lowering leg. Selecting this option removes only that last checked leg.
    """
    if not checked_enabled(getattr(node, 'place_finish_keep_torso_height_enabled', False)):
        return {}, home
    if (not getattr(node, 'delivery_evidence_enabled', False)
            or not getattr(node, 'table_scene_required', False)
            or not getattr(node, 'bin_scene_required', False)
            or direct_empty_home is None
            or getattr(node, '_active_place_scene_reference', None) is None):
        raise RuntimeError('raised PLACE finish requires its correlated checked direct return')
    identity = AttemptIdentity(**correlation)
    # Successful opening clears the pickup target/contact caches. The original
    # attempt-local PLACE guard retains its full identity until command finally.
    with getattr(node, '_lock', None) or nullcontext():
        guard = getattr(node, '_place_contact_guard', None)
        if (not isinstance(guard, PlaceContactGuard)
                or not identity.matches(guard.correlation) or guard.fault is not None
                or type(guard.started_ns) is not int or guard.started_ns <= 0):
            raise RuntimeError('raised PLACE finish attempt guard changed or faulted')
        if (getattr(node, '_held_book_corners', object()) is not None
                or getattr(node, '_target_book_model', object()) is not None
                or getattr(node, '_gripper_open_confirmed', False) is not True):
            raise RuntimeError('raised PLACE finish requires measured unloaded release')
    q = np.asarray(torso_ready, dtype=float)
    goal = np.asarray(home, dtype=float).copy()
    if (q.shape != (8,) or goal.shape != (8,)
            or not np.all(np.isfinite(q)) or not np.all(np.isfinite(goal))
            or not float(node.chain.lower[0]) <= q[0] <= float(node.chain.upper[0])):
        raise ValueError('raised PLACE finish requires a finite admitted torso-ready state')
    goal[0] = q[0]
    return {'keep_torso_height': True}, goal
