"""Fresh pre-send slope admission for optional faster position-only goals.

Caller holds the sensor lock through capture/check/publication, in the existing
command->sensor order. Goals are never modified. Evidence publication is separate
and must occur after locks release. This is not an acceleration/tracking proof.
"""
import math
from .motion_profiles import ARM_JOINTS


class ArmVelocityAdmissionRejected(RuntimeError):
    """The caller has not published this optional faster goal."""
    def __init__(self, reason, record=None):
        super().__init__(reason)
        self.record = {} if record is None else record


def _finite_values(values, count, label):
    try:
        # Bound iteration before copying malformed test/custom input.
        if len(values) != count:
            raise ArmVelocityAdmissionRejected(label)
        result=tuple(float(value) for value in values)
    except (TypeError,ValueError,OverflowError) as error:
        raise ArmVelocityAdmissionRejected(label) from error
    if not all(math.isfinite(value) for value in result):
        raise ArmVelocityAdmissionRejected(label)
    return result


def _duration_ns(duration):
    seconds,nanoseconds=duration.sec,duration.nanosec
    if (type(seconds) is not int or type(nanoseconds) is not int
            or not 0<=seconds<=2_147_483_647 or not 0<=nanoseconds<1_000_000_000):
        raise ArmVelocityAdmissionRejected('invalid serialized arm time')
    return seconds*1_000_000_000+nanoseconds


def _goal_record(goal,start_positions,velocity_limits):
    record={'record_complete':False}
    try:
        names=tuple(ARM_JOINTS)
        trajectory=goal.trajectory
        if len(trajectory.joint_names)!=len(names):
            raise ArmVelocityAdmissionRejected('unexpected arm joint order')
        record['joint_names']=list(trajectory.joint_names)
        record['header_stamp_ns']=_duration_ns(trajectory.header.stamp)
        record['start_positions']=list(_finite_values(start_positions,len(names),'invalid measured arm start'))
        record['velocity_limits']=list(_finite_values(velocity_limits,len(names),'invalid official arm limits'))
        record['point_count']=len(trajectory.points)
        if not 1<=record['point_count']<=256:
            raise ArmVelocityAdmissionRejected('invalid faster arm point count')
        record['points']=[]
        for point in trajectory.points:
            record['points'].append(dict(
                positions=list(_finite_values(point.positions,len(names),'invalid arm positions')),
                time_from_start_ns=_duration_ns(point.time_from_start),
                velocities_count=len(point.velocities),accelerations_count=len(point.accelerations),
                effort_count=len(point.effort)))
        record['record_complete']=True
        return record
    except ArmVelocityAdmissionRejected as error:
        error.record=record
        raise
    except (AttributeError,TypeError,ValueError,OverflowError) as error:
        raise ArmVelocityAdmissionRejected('malformed serialized faster arm goal',record) from error


def _check_record(record):
    names=tuple(ARM_JOINTS)
    try:
        if tuple(record['joint_names'])!=names:
            raise ArmVelocityAdmissionRejected('unexpected arm joint order')
        if record['header_stamp_ns']!=0:
            raise ArmVelocityAdmissionRejected('faster arm goal must start immediately')
        limits=record['velocity_limits']
        if any(value<=0 for value in limits):
            raise ArmVelocityAdmissionRejected('nonpositive official arm limit')
        previous,previous_ns=record['start_positions'],0
        maximum_ratio=0.
        for index,point in enumerate(record['points']):
            if point['velocities_count'] or point['accelerations_count'] or point['effort_count']:
                raise ArmVelocityAdmissionRejected('faster arm interpolation changed')
            delta_ns=point['time_from_start_ns']-previous_ns
            if delta_ns<=0:
                raise ArmVelocityAdmissionRejected('arm point times must strictly increase')
            seconds=delta_ns/1_000_000_000
            for name,first,last,limit in zip(names,previous,point['positions'],limits):
                speed=abs(last-first)/seconds
                if not math.isfinite(speed) or speed>limit:
                    raise ArmVelocityAdmissionRejected(
                        f'commanded arm velocity exceeds official limit: {name}, segment {index}')
                maximum_ratio=max(maximum_ratio,speed/limit)
            previous,previous_ns=point['positions'],point['time_from_start_ns']
        record['maximum_commanded_velocity_limit_ratio']=maximum_ratio
        return record
    except ArmVelocityAdmissionRejected as error:
        error.record=record
        raise


def check_serialized_arm_goal(goal,start_positions,velocity_limits):
    return _check_record(_goal_record(goal,start_positions,velocity_limits))


def require_arm_velocity_locked(node,goal):
    """No lock acquisition, waits, publication, file reads or node mutation."""
    record={}
    try:
        names=tuple(ARM_JOINTS)
        now_ns=node.get_clock().now().nanoseconds
        if type(now_ns) is not int or now_ns<=0:
            raise ArmVelocityAdmissionRejected('invalid faster arm clock')
        stamps=getattr(node,'_joint_stamps_ns',{})
        producer_stamps=[stamps.get(name) for name in names]
        positions=[node.joints.get(name,math.nan) for name in names]
        metadata=dict(evaluated_ros_ns=now_ns,producer_stamps_ns=producer_stamps,
                      official_urdf=getattr(node,'_faster_arm_velocity_urdf',None))
        record=_goal_record(goal,positions,getattr(node,'_faster_arm_velocity_limits',()))
        record.update(metadata)
        for name,stamp in zip(names,producer_stamps):
            if (type(stamp) is not int or stamp<=0
                    or not -50_000_000<=now_ns-stamp<=150_000_000):
                raise ArmVelocityAdmissionRejected(f'faster arm feedback stale: {name}',record)
        return _check_record(record)
    except ArmVelocityAdmissionRejected as error:
        if 'metadata' in locals():error.record.update(metadata)
        raise
    except (AttributeError,TypeError,ValueError,OverflowError) as error:
        raise ArmVelocityAdmissionRejected('malformed faster arm feedback',record) from error


def publish_arm_velocity_admission(node,record,*,command,admitted,reason=None):
    """Best effort exactly once per checked send; caller is outside both locks.

    admitted means the slope gate passed, never action acceptance or execution.
    Invalid inputs may have only a partial record, marked record_complete=False.
    """
    try:
        node._publish_status('arm_velocity_admission',command=command,admitted=bool(admitted),
                             reason=reason,**record)
    except Exception:
        pass
