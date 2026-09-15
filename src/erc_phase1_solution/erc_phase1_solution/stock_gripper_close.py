"""Explicit stock-simulator position diagnostic; no actuator/physics edits.

Contact evidence is not a pressure or attachment certificate. Numerical force
and master-effort amplitude policies are disabled only for this opt-in mode.
"""
import math
import time

from .adaptive_grasp import AdaptiveGraspEvidence


def enabled(node):
    return bool(getattr(node, 'stock_gripper_close_diagnostic_enabled', False))


# Stock measured-coordinate engineering allowance; public commands stay [0, .069].
# Raw sensor values remain in evidence and are never globally normalized.
MEASURED_POSITION_ROUNDOFF_M = 1e-6


def measured_position_in_range(position):
    return (math.isfinite(position)
            and -MEASURED_POSITION_ROUNDOFF_M <= position
            <= .069 + MEASURED_POSITION_ROUNDOFF_M)


def contact_evidence(left, right, feedback, now, *, maximum_velocity=.003):
    """Fresh exact-pair chronology supplied by caller, with actual force data.

No calibrated pressure minimum, target-error check or master-to-pad conversion.
The caller owns exact identity, arm geometry and final publication locking.
"""
    fields=dict(width=getattr(feedback,'position',math.nan),
        left_force=left[-1].force_newtons if left else math.nan,
        right_force=right[-1].force_newtons if right else math.nan,
        effort=getattr(feedback,'effort',math.nan),effort_delta=math.nan,
        left_samples=0,right_samples=0)
    def result(ok,reason): return AdaptiveGraspEvidence(ok,reason,**fields)
    if (type(now) is not int or now<=0 or feedback is None
            or type(feedback.stamp_ns) is not int
            or not all(math.isfinite(v) for v in (feedback.position,feedback.velocity,feedback.effort))
            or not 0<=now-feedback.stamp_ns<=150_000_000):
        return result(False,'invalid_gripper_feedback')
    if not measured_position_in_range(feedback.position):
        return result(False,'width_invalid')
    if abs(feedback.velocity)>maximum_velocity:
        return result(False,'joint_still_moving')
    runs=[]
    for index,side in enumerate((left,right)):
        previous=0
        for sample in side:
            if type(sample.stamp_ns) is not int or not previous<sample.stamp_ns<=now:
                return result(False,'malformed_force_history')
            if not math.isfinite(sample.force_newtons) or sample.force_newtons<0:
                return result(False,'invalid_contact_force')
            previous=sample.stamp_ns
        recent=[s for s in side if now-s.stamp_ns<=150_000_000]
        if not recent:return result(False,'unilateral_contact')
        if now-recent[-1].stamp_ns>75_000_000:return result(False,'unilateral_contact')
        run=[]
        for sample in recent:
            if run and sample.stamp_ns-run[-1].stamp_ns>75_000_000:run=[]
            run.append(sample)
        fields['left_samples' if index==0 else 'right_samples']=len(run)
        if len(run)<3:return result(False,'insufficient_force_samples')
        if run[-1].stamp_ns-run[0].stamp_ns<50_000_000:
            return result(False,'force_sample_span_too_short')
        runs.append(run)
    if abs(runs[0][-1].stamp_ns-runs[1][-1].stamp_ns)>50_000_000:
        return result(False,'bilateral_sample_skew')
    return result(True,'fresh_named_bilateral_contact')


class StockGripperClose:
    """One selected-target/1s command, short confirmation, no success hold."""
    def __init__(self,node):
        self.node=node
        self.target=float(getattr(node,'stock_gripper_close_target_position',0.))
        if not math.isfinite(self.target) or not 0.<=self.target<=.069:
            raise ValueError('stock close target must be finite and within public range [0, .069]')
        self.started_wall=time.monotonic()
        self.deadline=self.started_wall+15.
        self.last_clock=int(node.get_clock().now().nanoseconds)
        self.clock_progress_wall=self.started_wall
        self.command_start=None
        self.command_end=None
        self.active=True

    def before_publish(self,position,motion_seconds,wait_seconds):
        if self.command_start is not None or (position,motion_seconds,wait_seconds)!=(self.target,1.,1.2):
            raise RuntimeError('stock close permits exactly one selected-target/1s command')
        self.guard()
        self.command_start=int(self.node.get_clock().now().nanoseconds)
        self.command_end=self.command_start+1_000_000_000

    def guard(self):
        n=self.node;wall=time.monotonic();now=int(n.get_clock().now().nanoseconds)
        if n._cancel.is_set():raise RuntimeError('cancelled')
        fault=(n._adaptive_overload_reason() or getattr(n,'_adaptive_motion_halt_reason',None)
               or ('payload_robot_contact' if getattr(n,'_target_robot_contact_latched',False) else None))
        if fault:raise RuntimeError(str(fault))
        if now<self.last_clock:raise RuntimeError('clock_reversed')
        if now>self.last_clock:self.clock_progress_wall=wall
        if wall-self.clock_progress_wall>1.:raise RuntimeError('clock_stalled')
        self.last_clock=now
        if wall>=self.deadline:raise RuntimeError('stock_close_wall_timeout')
        feedback,error=n._adaptive_motion_feedback()
        if error or feedback is None or not math.isfinite(feedback.effort):
            raise RuntimeError(error or 'invalid_gripper_effort')
        return now

    def wait(self,duration):
        try:
            if self.command_start is None:raise RuntimeError('stock command absent')
            end=self.command_start+int(duration*1e9)
            while self.guard()<end:time.sleep(.002)
            return True
        except Exception as exc:
            self.node._hold_adaptive_gripper(str(exc))
            return False

    def run(self):
        n=self.node
        n._transport_lock_engaged=False
        n._clear_target_contact_samples()
        baseline=n._adaptive_effort_baseline(int(n.get_clock().now().nanoseconds))
        if not math.isfinite(baseline):raise RuntimeError('stock effort feedback unavailable')
        n._adaptive_effort_baseline_value=baseline
        n._stock_effort_peak=0.
        n._publish_status('gripper_close_diagnostic',control_mode='stock_public_position',
            target=self.target,motion_seconds=1.,force_amplitude_cutoff=None,
            effort_amplitude_cutoff=None,stock_actuator_limits_unchanged=True,
            wall_timeout_seconds=15.,contact_only=True)
        if not n._publish_adaptive_gripper_position(self.target,motion_seconds=1.,wait_seconds=1.2):
            return False,0.,False,False,False
        while True:
            now=self.guard()
            evidence,model,identity=n._adaptive_pressure_evidence(minimum_width=0.,baseline_effort=baseline)
            verified=evidence.verified and model is not None and identity is None
            if verified:
                # Normal result helper only engages a software state flag. It
                # sends no .015 lock or measured-q command on this branch.
                return n._publish_adaptive_close_result(evidence,verified=True,
                    reason='stock_named_contact_confirmed',close_stage='stock_public_position',
                    commanded_position=self.target,acquisition_position=evidence.width)
            if (now>=self.command_start+2_000_000_000 or identity is not None
                    or evidence.reason not in {'unilateral_contact','insufficient_force_samples',
                        'force_sample_span_too_short','bilateral_sample_skew','joint_still_moving'}):
                n._hold_adaptive_gripper(identity or evidence.reason)
                return n._publish_adaptive_close_result(evidence,verified=False,
                    reason=identity or evidence.reason,close_stage='stock_public_position',
                    commanded_position=self.target,acquisition_position=None)
            time.sleep(.002)
