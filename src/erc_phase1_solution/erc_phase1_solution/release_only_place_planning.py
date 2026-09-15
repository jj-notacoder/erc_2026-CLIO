"""Normal release-only planning capability; never an empty-return route."""
from dataclasses import dataclass
import json

import numpy as np

from .place_contact_guard import PlaceContactGuard
from .placement_scene_context import measured_scene_context
from .release_evidence import AttemptIdentity


def checked_enabled(value):
    if type(value) is not bool:
        raise ValueError('release_only_place_planning_enabled must be boolean')
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class Request:
    node: object
    identity: AttemptIdentity
    reference: object
    reference_bytes: str
    bin_scene: object
    bin_bytes: str
    table_scene: object
    table_bytes: str
    guard: PlaceContactGuard
    guard_epoch: int
    held: object
    held_bytes: bytes
    chains: tuple
    open_position: float

    def require_locked(self):
        """Caller owns node._lock; no waits, measurements or nested locks."""
        n = self.node
        if (not checked_enabled(getattr(n, 'release_only_place_planning_enabled', False))
                or getattr(n, 'place_finish_at_release_enabled', False) is not True
                or getattr(n, 'delivery_evidence_enabled', False) is not True
                or getattr(n, 'book_centered_place_enabled', False) is not True
                or getattr(n, 'table_scene_required', False) is not True
                or getattr(n, 'bin_scene_required', False) is not True
                or getattr(n, 'bin_clearance_timing_enabled', False)
                or n._cancel.is_set() or not n._busy
                or n._target_book_model != self.identity.target_model
                or n._active_place_scene_reference is not self.reference
                or n._selected_place_scene_reference is not self.reference
                or n._selected_place_bin_scene is not self.bin_scene
                or n._selected_place_table_scene is not self.table_scene
                or _json(self.reference) != self.reference_bytes
                or _json(self.bin_scene) != self.bin_bytes
                or _json(self.table_scene) != self.table_bytes
                or n._place_contact_guard is not self.guard
                or not self.identity.matches(self.guard.correlation)
                or self.guard.started_ns != self.guard_epoch or self.guard.fault is not None
                or getattr(n, '_payload_hazard_latched', None)
                or getattr(n, '_target_robot_contact_latched', False)
                or getattr(n, '_release_pose_fault_latched', None)
                or n._held_book_corners is not self.held
                or np.asarray(self.held, dtype=float).shape != (8, 3)
                or np.asarray(self.held, dtype=float).tobytes() != self.held_bytes
                or any(a is not b for a, b in zip((n.chain,n.right_chain,n.head_chain),self.chains))
                or n.gripper_open != self.open_position):
            raise RuntimeError('release_only_plan_attempt_scene_or_policy_changed')

    def require(self):
        with self.node._lock:
            self.require_locked()
        measured_scene_context(self.node, self.reference)
        with self.node._lock:
            self.require_locked()

    def endpoint(self, goal, scene):
        self.require()
        values = np.asarray(goal, dtype=float)
        if values.shape != (8,) or not np.all(np.isfinite(values)):
            raise ValueError('release-only endpoint must be finite')
        # This preserves the first unloaded sample of the original return leg;
        # only subsequent unused motion samples may be omitted.
        if not scene.sample(values, self.open_position, False):
            return None
        self.require()
        return Endpoint(self, tuple(float(v) for v in values))


@dataclass(frozen=True)
class Endpoint:
    request: Request
    goal: tuple

    def require(self, node, correlation, goal):
        if (node is not self.request.node or not self.request.identity.matches(correlation or {})
                or tuple(float(v) for v in goal) != self.goal):
            raise RuntimeError('release_only_endpoint_identity_changed')
        self.request.require()


@dataclass(frozen=True)
class OmittedReturn:
    """Non-iterable marker, deliberately neither None nor the direct[] result."""
    request: Request


def reject_return(value):
    if isinstance(value, OmittedReturn):
        raise RuntimeError('release_only_plan_has_no_return_or_recovery_route')


def capture(node, payload):
    if not checked_enabled(getattr(node, 'release_only_place_planning_enabled', False)):
        return None
    if not isinstance(payload, dict) or payload.get('completion_mode') != 'release_pose':
        raise RuntimeError('release_only_plan_requires_normal_release_mode')
    identity = AttemptIdentity(**{k:payload.get(k) for k in
                                 ('trial_id','placement_attempt_id','target_model')})
    with node._lock:
        guard = getattr(node, '_place_contact_guard', None)
        if (type(guard) is not PlaceContactGuard or type(guard.started_ns) is not int
                or guard.started_ns <= 0 or node._goal_handles
                or getattr(node, '_pending_retained_acceptances', ())
                or getattr(node, '_release_pose_owner', None) is not None):
            raise RuntimeError('release_only_plan_requires_idle_normal_attempt')
        reference = getattr(node, '_active_place_scene_reference', None)
        bin_scene = getattr(node, '_selected_place_bin_scene', None)
        table_scene = getattr(node, '_selected_place_table_scene', None)
        if not all(type(v) is dict for v in (reference,bin_scene,table_scene)):
            raise RuntimeError('release_only_plan_requires_registered_scene')
        held = node._held_book_corners
        request = Request(node,identity,reference,_json(reference),bin_scene,_json(bin_scene),
            table_scene,_json(table_scene),guard,guard.started_ns,held,
            np.asarray(held,dtype=float).tobytes(),(node.chain,node.right_chain,node.head_chain),node.gripper_open)
        request.require_locked()
    request.require()
    return request


def require_plan(request, plan, node, correlation):
    if type(request) is not Request:
        raise RuntimeError('release_only_plan_missing_request')
    proof = getattr(plan, 'release_only_endpoint', None)
    if (type(proof) is not Endpoint or proof.request is not request
            or type(plan.unloaded_home) is not OmittedReturn
            or plan.unloaded_home.request is not request
            or plan.empty_return_from_clearance is not None):
        raise RuntimeError('release_only_plan_missing_checked_endpoint')
    proof.require(node,correlation,plan.solutions[-1])
    return proof
