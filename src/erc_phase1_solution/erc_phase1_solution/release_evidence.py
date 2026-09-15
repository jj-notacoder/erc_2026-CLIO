"""Attempt-correlated placement operation evidence; no containment inference.

No ROS imports or motion commands. One owner lock must serialize all calls.
Physical inside verification remains a separate observation/reporting claim.
"""
from dataclasses import dataclass
import math
import re


def stamp_valid(value):
    return type(value) is int and value > 0


@dataclass(frozen=True)
class AttemptIdentity:
    trial_id: str
    placement_attempt_id: str
    target_model: str

    def __post_init__(self):
        if (not all(isinstance(v,str) and bool(v) for v in
                    (self.trial_id,self.placement_attempt_id,self.target_model))
                or re.fullmatch(r'book_col_\d+_row_\d+_(red|green|yellow|blue)',self.target_model) is None):
            raise ValueError('Concrete trial/attempt/target identity required')

    def matches(self, payload):
        return all(payload.get(key)==getattr(self,key) for key in
                   ('trial_id','placement_attempt_id','target_model'))


class ReleaseEvidence:
    """Bounded time history works at10Hz or500Hz without inventing frames."""
    def __init__(self, identity):
        self.identity=identity
        self.open_epoch=None
        self.return_measurement=None
        self.contact_stamps=set()
        self.verification_start=None
        self.last_ros=None
        self.fault=None
        self.pending_milestones={}

    def _clock(self, now_ns):
        if not stamp_valid(now_ns) or (self.last_ros is not None and now_ns<self.last_ros):
            self.fault=self.fault or 'invalid_or_reversed_clock'
            return False
        self.last_ros=now_ns
        return True

    def opening(self,payload,now_ns):
        if not self.identity.matches(payload):return False
        if not self._clock(now_ns):return False
        stamp=payload.get('producer_stamp_ns')
        if (payload.get('event')=='placement_open_measured' and payload.get('verified') is True
                and stamp_valid(stamp) and 0<stamp-now_ns<=100_000_000):
            self.pending_milestones['opening']=dict(payload)
            return False
        if (payload.get('event')!='placement_open_measured'
                or payload.get('verified') is not True or not stamp_valid(stamp)
                or not 0<=now_ns-stamp<=250_000_000):
            self.fault=self.fault or 'invalid_measured_open'
            return False
        if self.open_epoch is None:self.open_epoch=stamp
        elif stamp!=self.open_epoch:
            # Repeated data cannot silently move release forward and erase a
            # contact/feedback failure from the original attempt.
            self.fault=self.fault or 'release_epoch_changed'
            return False
        self.contact_stamps={s for s in self.contact_stamps if s>self.open_epoch}
        return True

    def hand_return(self,payload,now_ns):
        if not self.identity.matches(payload):return False
        if not self._clock(now_ns):return False
        stamp=payload.get('producer_stamp_ns')
        if (payload.get('event')=='placement_hand_return_measured' and payload.get('verified') is True
                and stamp_valid(stamp) and -250_000_000<=stamp-now_ns<=100_000_000
                and (stamp>now_ns or self.open_epoch is None)):
            self.pending_milestones['hand_return']=dict(payload)
            return False
        if (payload.get('event')!='placement_hand_return_measured'
                or self.open_epoch is None or not stamp_valid(stamp)
                or not self.open_epoch<stamp<=now_ns or now_ns-stamp>250_000_000
                or payload.get('verified') is not True):
            self.fault=self.fault or 'hand_return_unverified'
            return False
        self.return_measurement=dict(payload)
        return True

    def contact(self,stamp_ns,now_ns,*,exact_target_bin_pair):
        if not self._clock(now_ns):return
        if not exact_target_bin_pair:return
        if not stamp_valid(stamp_ns) or not -100_000_000<=now_ns-stamp_ns<=250_000_000:
            return
        # Buffer a short period before opening-status delivery too: Contacts
        # may arrive first. Their producer times, not delivery order, decide.
        self.contact_stamps={s for s in self.contact_stamps if s>=now_ns-750_000_000}
        if self.open_epoch is None or stamp_ns>self.open_epoch:
            self.contact_stamps.add(stamp_ns)
        if len(self.contact_stamps)>1024:
            self.fault=self.fault or 'contact_history_rate_overflow'
            self.contact_stamps.clear()

    def motion_terminal(self,payload,now_ns,wall_seconds):
        if not self.identity.matches(payload):return False
        if not self._clock(now_ns):return False
        if payload.get('command')!='place' or payload.get('event')!='succeeded':return False
        if not isinstance(wall_seconds,(int,float)) or not math.isfinite(wall_seconds):
            self.fault=self.fault or 'invalid_wall_clock';return False
        if self.verification_start is None:
            self.verification_start=(now_ns,float(wall_seconds))
        return True

    def invalidate(self,reason):
        self.fault=self.fault or str(reason)

    def evaluate(self,now_ns,wall_seconds):
        self._clock(now_ns)
        for kind in ('opening','hand_return'):
            payload=self.pending_milestones.get(kind)
            if payload is not None and payload['producer_stamp_ns']<=now_ns:
                del self.pending_milestones[kind]
                getattr(self,kind)(payload,now_ns)
        result=dict(measured_open=self.open_epoch is not None,
                    hand_return_measured=self.return_measurement is not None,
                    hand_clear_verified=False,post_release_contact_verified=False,
                    containment_verified=False,physical_delivery_verified=False,
                    physical_inside_verified=None, operation_completed=False,
                    success_scope=None,delivery_outcome=None,
                    status='pending',reason=self.fault)
        if self.fault:result['status']='invalid';return result
        # The terminal starts one bounded verification period even when a
        # required measurement never arrives. Missing evidence cannot bypass
        # clock/deadline validation or silently renew that period.
        if self.verification_start is not None:
            ros_start,wall_start=self.verification_start
            if (not isinstance(wall_seconds,(int,float)) or not math.isfinite(wall_seconds)
                    or wall_seconds<wall_start):
                result.update(status='invalid',reason='invalid_wall_clock');return result
            if now_ns-ros_start>10_000_000_000 or wall_seconds-wall_start>30.:
                result.update(status='expired',reason='verification_deadline_expired');return result
        if self.open_epoch is None:return result
        stamps=sorted(s for s in self.contact_stamps
                      if self.open_epoch<s<=now_ns and s>=now_ns-750_000_000)
        run=[]
        for value in stamps:
            if run and value-run[-1]>125_000_000:run=[]
            run.append(value)
        result['post_release_contact_verified']=bool(
            len(run)>=3 and run[-1]-run[0]>=500_000_000
            and now_ns-run[-1]<=150_000_000)
        result['contact_span_seconds']=(run[-1]-run[0])/1e9 if run else 0.
        result['contact_samples']=len(run)
        if self.verification_start is None:return result
        # A measured return is a milestone, not a current scene certificate or
        # proof the book lies inside the bin. No success boolean is fabricated.
        if result['post_release_contact_verified'] and self.return_measurement is not None:
            result.update(status='operation_completed',reason=None,operation_completed=True,
                          success_scope='placement_operation_completed',
                          delivery_outcome='released_with_bin_contact')
        return result


def measured_pose_is_open(raw,odom,goal,now_ns,joint_names,*,pending_negative_check=False,
                          open_position=.069):
    """Actuated position/velocity evidence; no passive gap or bin claim."""
    stamp=raw.get('producer_stamp_ns') if raw else None
    earliest=-100_000_000 if pending_negative_check else 0
    if not stamp_valid(stamp) or not earliest<=now_ns-stamp<=150_000_000:
        return False,'raw_joint_feedback_unconfirmed'
    q,v=raw.get('positions',{}),raw.get('velocities',{})
    names=(*joint_names,'gripper_left_finger_joint')
    if any(not math.isfinite(q.get(n,math.nan)) or not math.isfinite(v.get(n,math.nan)) for n in names):
        return False,'joint_feedback_invalid'
    if abs(q['gripper_left_finger_joint']-open_position)>.0005 or abs(v['gripper_left_finger_joint'])>.0001:
        return False,'master_not_measured_open'
    if len(goal)!=len(joint_names):return False,'goal_invalid'
    for index,name in enumerate(joint_names):
        tolerance=.001 if name=='torso_lift_joint' else .002
        if abs(q[name]-goal[index])>tolerance or abs(v[name])>.001:
            return False,'arm_not_at_stationary_checked_endpoint'
    if (not odom or not stamp_valid(odom.get('stamp_ns'))
            or not earliest<=now_ns-odom['stamp_ns']<=150_000_000
            or not 0<=odom.get('linear_speed',math.inf)<=.005
            or not 0<=odom.get('angular_speed',math.inf)<=.008):
        return False,'base_not_measured_stationary'
    return True,'measured_open_at_checked_endpoint'
