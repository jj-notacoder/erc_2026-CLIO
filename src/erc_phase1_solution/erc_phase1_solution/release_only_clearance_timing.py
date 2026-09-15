"""Default-off timing for the sole accepted release-only clearance connector.

The route capability is inactive during transition motion. A separate publication
owner is created only after every earlier leg retained its payload and the
original transition-stop owner completed. No publisher, planning or retry lives
here. The old clearance mode and transition-stop exclusions remain unchanged.
"""
from dataclasses import dataclass
import math
import time

from .arm_velocity_admission import ArmVelocityAdmissionRejected
from .bin_clearance_timing import ClearanceTiming, require_endpoint as original_endpoint
from .motion_profiles import ARM_JOINTS, IK_JOINTS
from .place_transition_stop import Qualification
from .release_only_place_planning import Endpoint, Request


def checked_enabled(value):
    if type(value) is not bool:
        raise ValueError('release_only_clearance_timing_enabled must be Boolean')
    return value


def _route(legs):
    route = []
    for q, duration, phase in legs:
        if (type(duration) is bool or phase not in ('bin_transition', 'bin_clearance', 'bin_approach')
                or len(q) != len(IK_JOINTS) or not all(math.isfinite(float(v)) for v in q)):
            raise ArmVelocityAdmissionRejected('release-only clearance original route invalid')
        expected = {'bin_transition': .8, 'bin_clearance': 2.8, 'bin_approach': .65}[phase]
        if float(duration) != expected:
            raise ArmVelocityAdmissionRejected('release-only clearance original duration changed')
        route.append((tuple(float(v) for v in q), float(duration), phase))
    phases = [x[2] for x in route]
    if phases.count('bin_clearance') != 1:
        raise ArmVelocityAdmissionRejected('release-only clearance requires one connector')
    index = phases.index('bin_clearance')
    if (index < 2 or index >= len(route)-1 or any(x != 'bin_transition' for x in phases[:index])
            or any(x != 'bin_approach' for x in phases[index+1:])):
        raise ArmVelocityAdmissionRejected('release-only clearance normal route required')
    return tuple(route), index


def normal_options(node, correlation, endpoint, legs, transition_stop, arm_speed_scale):
    if not checked_enabled(getattr(node, 'release_only_clearance_timing_enabled', False)):
        return {}
    if (type(endpoint) is not Endpoint or type(endpoint.request) is not Request
            or type(transition_stop) is not Qualification or transition_stop.node is not node
            or arm_speed_scale != 2.0 or isinstance(arm_speed_scale, bool)):
        raise ArmVelocityAdmissionRejected('typed release-only endpoint and cap2 stop required')
    route, index = _route(legs)
    if transition_stop.route != route or endpoint.goal != route[-1][0]:
        raise ArmVelocityAdmissionRejected('release-only clearance accepted route changed')
    endpoint.require(node, correlation, route[-1][0])
    plan = Plan(node, endpoint, transition_stop, route, index)
    with node._lock:
        plan._context_locked()
        if transition_stop.active or transition_stop.qualified or transition_stop.published:
            raise ArmVelocityAdmissionRejected('release-only clearance needs fresh transition capability')
    return dict(release_clearance=plan)


@dataclass
class Plan:
    node: object
    endpoint: Endpoint
    transition: Qualification
    route: tuple
    index: int
    started: bool = False
    completed_legs: int = 0
    admission: object = None

    def _context_locked(self):
        n = self.node
        if (not checked_enabled(getattr(n, 'release_only_clearance_timing_enabled', False))
                or getattr(n, 'bin_clearance_timing_enabled', False)
                or getattr(n, 'place_transition_stop_enabled', False) is not True
                or n.loaded_place_speed_scale_cap != 2.0 or n.additional_arm_time_scale != 2.0
                or n._cancel.is_set() or n._goal_handles
                or getattr(n, '_pending_retained_acceptances', ())
                or getattr(n, '_release_pose_owner', None) is not None
                or getattr(n, '_initial_stow_serial_owner', None) is not None
                or getattr(n, '_retention_probe_active', False)
                or getattr(n, '_payload_monitor_enabled', False) is not True
                or getattr(n, '_gripper_open_confirmed', False)):
            raise ArmVelocityAdmissionRejected('release-only clearance policy or ownership changed')
        for key in ('_payload_hazard_latched', '_held_grip_sensor_fault', '_raw_contacts_first_failure',
                    '_release_pose_fault_latched'):
            if getattr(n, key, None) is not None:
                raise ArmVelocityAdmissionRejected(str(getattr(n, key)))
        self.endpoint.request.require_locked()

    def _completed_stop_locked(self):
        t, n = self.transition, self.node
        if (type(t) is not Qualification or t.node is not n or t.route != self.route
                or not t.qualified or not t.published or t.active
                or getattr(n, '_place_transition_stop_owner', None) is not None
                or getattr(n, '_delivery_measurement_active', False)
                or t.reference is not self.endpoint.request.reference
                or t.epoch != getattr(n, '_contact_epoch', None)
                or t.target != self.endpoint.request.identity.target_model):
            raise ArmVelocityAdmissionRejected('original transition stop not completed and closed')

    def require_execution(self, node, legs, command, *, arm_speed_scale, leg_offset,
                          fresh_retention_phases, initial_pressure_gate, clearance_timing,
                          withdrawal_timing, withdrawal_speed_scale, transition_stop):
        if (node is not self.node or command != 'place' or _route(legs) != (self.route, self.index)
                or arm_speed_scale != 2.0 or isinstance(arm_speed_scale, bool) or leg_offset != 0
                or fresh_retention_phases or initial_pressure_gate is not None
                or clearance_timing is not None or withdrawal_timing is not None
                or withdrawal_speed_scale != 1.0 or isinstance(withdrawal_speed_scale, bool)
                or transition_stop is not self.transition or self.started):
            raise ArmVelocityAdmissionRejected('release-only clearance execution scope changed')
        with node._lock:
            self._context_locked()
            if getattr(node, '_release_only_clearance_owner', None) is not None:
                raise ArmVelocityAdmissionRejected('release-only clearance already owned')
            self.started = True

    def retained(self, index):
        # Called only after the actual sender succeeds and original retention passes.
        if not self.started or index != self.completed_legs:
            raise ArmVelocityAdmissionRejected('release-only clearance completion order changed')
        if index == 1:
            with self.node._lock:
                self._completed_stop_locked()
        self.completed_legs += 1

    def admit(self, index, solution, duration, phase):
        if (not self.started or index != self.index or self.completed_legs != index
                or self.admission is not None or duration != 2.8 or phase != 'bin_clearance'
                or tuple(float(v) for v in solution) != self.route[index][0]):
            raise ArmVelocityAdmissionRejected('release-only clearance connector boundary changed')
        n = self.node
        self.endpoint.require(n, self.endpoint.request.guard.correlation, self.route[-1][0])
        reason = n._payload_hazard_reason(max_age=min(.15, float(n.grasp_contact_max_age)))
        if reason is not None:
            raise ArmVelocityAdmissionRejected('release-only clearance payload hazard: '+str(reason))
        with n._lock:
            self._context_locked()
            self._completed_stop_locked()
            if getattr(n, '_release_only_clearance_owner', None) is not None:
                raise ArmVelocityAdmissionRejected('release-only clearance already owned')
            token = Admission(self, time.monotonic())
            n._release_only_clearance_owner = token
            self.admission = token
        return token


@dataclass
class Admission:
    plan: Plan
    started_wall: float
    closed: bool = False
    published: bool = False

    def require_publication_locked(self, node, goal, command, leg_offset):
        try:
            self._require_publication_locked(node, goal, command, leg_offset)
        except ArmVelocityAdmissionRejected:
            raise
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            # This call precedes pending registration; preserve that distinction
            # for the original sender even when a typed release request vetoes.
            raise ArmVelocityAdmissionRejected('release-only clearance publication rejected: '+str(error)) from error

    def _require_publication_locked(self, node, goal, command, leg_offset):
        p = self.plan
        if (node is not p.node or command != 'place' or leg_offset != p.index
                or self.closed or self.published or p.admission is not self
                or getattr(node, '_release_only_clearance_owner', None) is not self
                or tuple(goal.trajectory.joint_names) != tuple(ARM_JOINTS)
                or len(goal.trajectory.points) != 1
                or tuple(goal.trajectory.points[0].positions) != p.route[p.index][0][1:]):
            raise ArmVelocityAdmissionRejected('release-only clearance publication scope changed')
        stamp = goal.trajectory.points[0].time_from_start
        if (type(stamp.sec) is not int or type(stamp.nanosec) is not int
                or not 0 <= stamp.nanosec < 1_000_000_000
                or not 350_000_000 <= stamp.sec*1_000_000_000+stamp.nanosec <= 700_000_000):
            raise ArmVelocityAdmissionRejected('release-only clearance serialized timing changed')
        p._context_locked()
        p._completed_stop_locked()
        self.published = True

    def close(self):
        with self.plan.node._lock:
            if getattr(self.plan.node, '_release_only_clearance_owner', None) is self:
                self.plan.node._release_only_clearance_owner = None
            self.closed = True

    def require_endpoint(self):
        p, n = self.plan, self.plan.node
        if not self.closed or not self.published:
            raise ArmVelocityAdmissionRejected('release-only clearance did not complete publication')
        with n._lock:
            p._context_locked()
            p._completed_stop_locked()
            if getattr(n, '_release_only_clearance_owner', None) is not None:
                raise ArmVelocityAdmissionRejected('release-only clearance endpoint owner changed')
        req = p.endpoint.request
        original = ClearanceTiming(req.identity, req.guard, req.reference, p.index, p.route[p.index][0])
        original_endpoint(n, original, p.index)
        req.require()
        n._publish_status('release_only_clearance_endpoint_verified', command='place',
            leg=p.index, nominal_watchdog_seconds=2.8, timing_input_seconds=1.4,
            elapsed_wall_seconds=time.monotonic()-self.started_wall,
            original_retained_endpoint_checked=True, stationary_verified=False,
            **req.guard.correlation)
