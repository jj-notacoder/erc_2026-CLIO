"""Default-off early empty-head timing; original sender and camera gates remain."""
from dataclasses import dataclass
import hashlib
import math
import re
import time
import uuid
import xml.etree.ElementTree as ET

from .motion_profiles import IK_JOINTS, RIGHT_ARM_JOINTS

HEAD = ('head_1_joint', 'head_2_joint')
BODY = (*IK_JOINTS, *RIGHT_ARM_JOINTS)
ID = 'empty_head_timing_id'
EPOCH = 'empty_head_not_before_ns'


def checked_enabled(value):
    if type(value) is not bool:
        raise ValueError('empty_head_timing_enabled must be Boolean')
    return value


@dataclass(frozen=True)
class HeadLimits:
    lower: tuple
    upper: tuple
    velocity: tuple
    urdf: str
    sha256: str


def load_limits(path):
    raw = path.read_bytes()
    root = ET.fromstring(raw)
    values = []
    for name in HEAD:
        joints = [j for j in root.findall('joint') if j.get('name') == name]
        if len(joints) != 1 or joints[0].get('type') != 'revolute':
            raise ValueError('official head joint is missing or ambiguous')
        limit = joints[0].find('limit')
        row = tuple(float(limit.get(key)) for key in ('lower', 'upper', 'velocity'))
        if not all(math.isfinite(v) for v in row) or row[0] >= row[1] or row[2] != 3.0:
            raise ValueError('expected official bounded 3rad/s head joints')
        values.append(row)
    return HeadLimits(tuple(r[0] for r in values), tuple(r[1] for r in values),
        tuple(r[2] for r in values), str(path), hashlib.sha256(raw).hexdigest())


def identity(payload):
    trial, request = (payload or {}).get('trial_id'), (payload or {}).get(ID)
    if (type(trial) is not str or re.fullmatch('[0-9a-f]{12}', trial) is None
            or type(request) is not str or re.fullmatch('[0-9a-f]{32}', request) is None):
        raise RuntimeError('empty head request identity invalid')
    return {'trial_id': trial, ID: request}


def command_fields(payload):
    return identity(payload) if ID in (payload or {}) else {}


def prepare(node, command, payload):
    if ID not in (payload or {}):
        return None
    if not checked_enabled(getattr(node, 'empty_head_timing_enabled', False)):
        raise RuntimeError('empty head timing request requires explicit opt-in')
    target = {'look_markers': (0., .20), 'look_books': (0., float(node.book_overview_tilt))}
    target.update({f'look_book_row_{i}': (0., float(v)) for i, v in enumerate(node.book_row_tilts, 1)})
    if command not in target or (payload or {}).get('empty_head_before_pick') is not True:
        raise RuntimeError('empty head command is outside initial scope')
    request=identity(payload)
    with node._lock:
        phase=getattr(node,'_empty_head_initial_phase',0)
        expected=(command=='look_markers' if phase==0 else command=='look_books' if phase==1
                  else command.startswith('look_book_row_') if phase==2 else False)
        if (not expected or getattr(node,'_empty_head_pick_started',False)
                or (request['trial_id'],request[ID]) in getattr(node,'_empty_head_seen_requests',set())):
            raise RuntimeError('empty head initial sequence/request invalid')
        node._empty_head_initial_phase=phase+1
        seen=getattr(node,'_empty_head_seen_requests',set())
        seen.add((request['trial_id'],request[ID]));node._empty_head_seen_requests=seen
    return EmptyHeadOwner(node, command, request, target[command])


class EmptyHeadOwner:
    def __init__(self, node, command, request, target):
        self.node, self.command, self.identity, self.target = node, command, request, tuple(target)
        self.limits = node._empty_head_timing_limits
        if type(self.limits) is not HeadLimits:
            raise RuntimeError('empty head official limits unavailable')
        self.token = object()
        self.acceptance = self.handle = self.result_future = None
        self.last_ros = None; self.last_stamps = {}
        self.last_odom_stamp = None
        self.start = self.base = self.parked = None
        self.duration_ns = None
        self.dispatched = self.late_registered = False
        self.completion = None
        self.head_chain = node.head_chain
        self.cancel = node._cancel

    def _snapshot_locked(self, *, stationary=False):
        n = self.node
        now = int(n.get_clock().now().nanoseconds)
        if self.last_ros is not None and now < self.last_ros:
            raise RuntimeError('empty head clock regressed')
        self.last_ros = now
        if (getattr(n, '_empty_head_timing_owner', None) is not self
                or not checked_enabled(n.empty_head_timing_enabled)
                or n._cancel is not self.cancel or n._cancel.is_set()
                or getattr(n, '_empty_head_pick_started', False)
                or n._empty_head_timing_limits is not self.limits or n.head_chain is not self.head_chain
                or getattr(n, '_held_book_corners', None) is not None
                or getattr(n, '_active_place_scene_reference', None) is not None
                or getattr(n, '_place_contact_guard', None) is not None
                or getattr(n, '_empty_torso_planning_owner', None) is not None
                or getattr(n, '_head_return_state', None) is not None
                or getattr(n, '_payload_hazard_latched', None) is not None
                or getattr(n, '_held_grip_sensor_fault', None) is not None
                or getattr(n, '_raw_contacts_first_failure', None) is not None
                or getattr(n, '_target_robot_contact_latched', False)
                or getattr(n, '_empty_arm_contact_latched', False)
                or getattr(n, '_adaptive_close_active', False)
                or getattr(n, '_transport_lock_engaged', False)
                or getattr(n, '_retention_probe_active', False)
                or getattr(n, '_empty_arm_motion_active', False)
                or getattr(n, '_empty_pickup_geometry_active', False)
                or getattr(n, '_pickup_parallel_geometry_active', False)):
            raise RuntimeError('empty head ownership or physical context invalid')
        positions, velocities, stamps = {}, {}, {}
        for name in (*BODY, *HEAD):
            q = n.joints.get(name); stamp = n._joint_stamps_ns.get(name)
            if (q is None or not math.isfinite(q) or type(stamp) is not int
                    or not -50_000_000 <= now-stamp <= 150_000_000
                    or stamp < self.last_stamps.get(name, stamp)):
                raise RuntimeError('empty head stale/invalid/regressed joint:' + name)
            positions[name], stamps[name] = float(q), stamp
            self.last_stamps[name] = stamp
        for i, name in enumerate(HEAD):
            v = n._joint_velocities.get(name)
            if (v is None or not math.isfinite(v)
                    or not self.limits.lower[i] <= positions[name] <= self.limits.upper[i]
                    or not self.limits.lower[i] <= self.target[i] <= self.limits.upper[i]
                    or (stationary and abs(v) > 1e-4)):
                raise RuntimeError('empty head position/velocity admission failed')
            velocities[name] = float(v)
        odom = getattr(n, '_staging_odom', None)
        if not isinstance(odom, dict) or type(odom.get('stamp_ns')) is not int:
            raise RuntimeError('empty head odometry missing')
        if self.last_odom_stamp is not None and odom['stamp_ns'] < self.last_odom_stamp:
            raise RuntimeError('empty head odometry producer regressed')
        self.last_odom_stamp = odom['stamp_ns']
        pose = tuple(odom.get('pose', ()))
        speeds = (odom.get('linear_speed'), odom.get('angular_speed'))
        if (len(pose) != 3 or not all(type(v) in (int, float) or hasattr(v, 'item') for v in pose)
                or not all(math.isfinite(v) for v in (*pose, *speeds))
                or not -50_000_000 <= now-odom['stamp_ns'] <= 150_000_000
                or not 0 <= speeds[0] <= .005 or not 0 <= speeds[1] <= .008):
            raise RuntimeError('empty head base is not fresh/stopped')
        if self.base is not None:
            yaw = math.atan2(math.sin(pose[2]-self.base[2]), math.cos(pose[2]-self.base[2]))
            if math.hypot(pose[0]-self.base[0], pose[1]-self.base[1]) > .002 or abs(yaw) > .005:
                raise RuntimeError('empty head base moved')
            if any(abs(positions[name]-q) > (.002 if name == IK_JOINTS[0] else .003)
                   for name, q in self.parked.items()):
                raise RuntimeError('empty head parked body moved')
        return dict(now=now, positions=positions, velocities=velocities, stamps=stamps, base=pose)

    def check(self):
        with self.node._lock:
            return self._snapshot_locked()

    def admit_locked(self, client, goal):
        n = self.node
        if (client is not n.head_client or self.dispatched or n._goal_handles
                or n._pending_retained_acceptances or tuple(goal.trajectory.joint_names) != HEAD
                or len(goal.trajectory.points) != 1):
            raise RuntimeError('empty head sender ownership invalid')
        point = goal.trajectory.points[0]
        if (tuple(point.positions) != self.target or point.velocities or point.accelerations or point.effort
                or point.time_from_start.sec != 1 or point.time_from_start.nanosec != 200_000_000):
            raise RuntimeError('empty head original goal changed')
        snapshot = self._snapshot_locked(stationary=True)
        self.start = tuple(snapshot['positions'][name] for name in HEAD)
        self.base = snapshot['base']; self.parked = {name:snapshot['positions'][name] for name in BODY}
        seconds = max(.6, *(abs(q-p)/(.5*v) for q,p,v in zip(self.target,self.start,self.limits.velocity)))
        ns = math.ceil(seconds*1_000_000_000)
        if not 600_000_000 <= ns <= 1_200_000_000:
            raise RuntimeError('empty head delta cannot satisfy bounded timing')
        point.time_from_start.sec, point.time_from_start.nanosec = divmod(ns, 1_000_000_000)
        self.duration_ns = ns
        n._pending_retained_acceptances.add(self.token)
        self.dispatched = True

    def sent_locked(self, future):
        self.acceptance = future

    def late(self):
        if self.acceptance is not None and not self.late_registered:
            self.late_registered = True
            self.acceptance.add_done_callback(lambda f:self.node._cancel_late_retained_goal(f,self.token))

    def wait_acceptance(self, future):
        deadline = time.monotonic()+self.node.timeout
        try:
            while not future.done():
                self.check()
                if time.monotonic() >= deadline:raise TimeoutError('empty head acceptance timeout')
                time.sleep(.02)
            handle = future.result()
            if handle is None or type(handle.accepted) is not bool:
                raise RuntimeError('empty head acceptance invalid')
            with self.node._lock:
                self.handle = handle
                if not handle.accepted:self.node._pending_retained_acceptances.discard(self.token)
            self.check()
            return handle
        except BaseException:
            self.late();raise

    def accepted_locked(self, handle, future):
        if handle is not self.handle or self.result_future is not None:
            raise RuntimeError('empty head accepted identity changed')
        self.result_future = future
        self.node._pending_retained_acceptances.discard(self.token)

    def terminal(self, future):
        self.check()
        if future is not self.result_future or not future.done() or not self.node._valid_retained_terminal_result(future.result()):
            raise RuntimeError('empty head terminal action unresolved')

    def cancel_owned(self):
        if self.handle is None:self.late();return
        if not self.handle.accepted:return
        try:
            try:self.handle.cancel_goal_async()
            except Exception:pass
            if self.result_future is None:self.result_future=self.handle.get_result_async()
            f=self.result_future
            if not (f.done() and self.node._valid_retained_terminal_result(f.result())):
                self.node._cancel_retained_goal_and_confirm(self.handle,f)
            if f.done() and self.node._valid_retained_terminal_result(f.result()):
                with self.node._lock:
                    if self.handle in self.node._goal_handles:self.node._goal_handles.remove(self.handle)
                    self.node._pending_retained_acceptances.discard(self.token)
        except Exception:pass

    def finish(self):
        n=self.node;started=int(n.get_clock().now().nanoseconds)
        wall=time.monotonic()+min(n.timeout,8.)
        previous=None
        while time.monotonic()<wall:
            with n._lock:
                snap=self._snapshot_locked()
                if snap['now']-started >= 2_000_000_000:raise TimeoutError('empty head endpoint ROS timeout')
                stamps=tuple(snap['stamps'][name] for name in HEAD)
                stopped=all(abs(snap['positions'][name]-target)<=1e-6
                    and abs(snap['velocities'][name])<=1e-4 and snap['stamps'][name]>started
                    for name,target in zip(HEAD,self.target))
                if stopped and previous is not None and all(a>b for a,b in zip(stamps,previous)):
                    self.completion=dict(self.identity,empty_head_measured_stop=True,
                        empty_head_completed_ros_ns=snap['now'],empty_head_action_return_ros_ns=started,
                        empty_head_producer_stamps_ns=list(stamps),empty_head_target=list(self.target),
                        empty_head_measured_positions=[snap['positions'][name] for name in HEAD],
                        empty_head_measured_velocities=[snap['velocities'][name] for name in HEAD],
                        empty_head_duration_ns=self.duration_ns)
                    n._empty_head_timing_owner=None
                    return True
                previous=stamps if stopped else None
            time.sleep(.02)
        raise TimeoutError('empty head measured endpoint timeout')


def run(node, owner, pan, tilt):
    if type(owner) is not EmptyHeadOwner or owner.node is not node or (pan,tilt)!=owner.target:
        raise RuntimeError('empty head call identity changed')
    with node._adaptive_command_guard():
        with node._lock:
            if getattr(node,'_empty_head_timing_owner',None) is not None:
                raise RuntimeError('empty head owner already active')
            node._empty_head_timing_owner=owner
    try:
        if not node._follow(node.head_client,HEAD,[pan,tilt],1.2,head_preflight=True,empty_head_owner=owner):
            raise RuntimeError('empty head original action failed')
        return owner.finish()
    except BaseException:
        node._last_completed_head_target=None
        node._cancel.set();owner.cancel_owned()
        # Failed owners remain command blockers even after the action terminates.
        raise


def mission_request(manager, command, fields):
    if not checked_enabled(getattr(manager,'empty_head_timing_enabled',False)):return
    if command=='pick':manager._empty_head_request=None;return
    allowed=(manager.state=='INIT_STOW' and command=='look_markers'
        or manager.state=='NAVIGATE_SHELF' and command=='look_books'
        or manager.state=='ALIGN_BOOK' and command==f'look_book_row_{manager.detected_row}')
    if not allowed or manager.pick_attempts!=0:return
    request=dict(trial_id=manager.trial_id,empty_head_timing_id=uuid.uuid4().hex,
        command=command,mode='markers' if command=='look_markers' else 'books',epoch=None)
    manager._empty_head_request=request
    manager.marker_cloud=None;manager.book_point=None
    manager._perception_mode('idle')
    fields.update({ID:request[ID],'empty_head_before_pick':True})


def mission_status(manager, payload):
    request=getattr(manager,'_empty_head_request',None)
    if request is None or payload.get('command')!=request['command']:return False
    if any(payload.get(k)!=request[k] for k in ('trial_id',ID)):
        manager._log('empty_head_status_rejected',reason='request_mismatch');return True
    if payload.get('event')=='succeeded':
        now=int(manager.get_clock().now().nanoseconds)
        stamp=payload.get('empty_head_completed_ros_ns');returned=payload.get('empty_head_action_return_ros_ns')
        producers=payload.get('empty_head_producer_stamps_ns',())
        if (payload.get('empty_head_measured_stop') is not True or type(stamp) is not int
                or type(returned) is not int or not 0<returned<=stamp<=now
                or len(producers)!=2 or any(type(v) is not int or not returned<v<=stamp+50_000_000 for v in producers)):
            manager._log('empty_head_status_rejected',reason='measured_completion_invalid');return True
        request['epoch']=stamp
    return False


def mission_mode(manager, mode, fields):
    request=getattr(manager,'_empty_head_request',None)
    if request is None:return
    if mode=='idle':
        fields[ID]=request[ID];fields['empty_head_expected_mode']=request['mode'];return
    if mode==request['mode']:
        if request['epoch'] is None:raise RuntimeError('empty head vision preceded measured completion')
        now=int(manager.get_clock().now().nanoseconds)
        if now<request['epoch']:raise RuntimeError('empty head camera clock regressed')
        request['epoch']=now
        fields.update({ID:request[ID],EPOCH:now})


def mission_observation_current(manager, mode, message):
    request=getattr(manager,'_empty_head_request',None)
    if request is None:return True
    if mode!=request['mode'] or request['epoch'] is None:return False
    stamp=message.header.stamp;value=int(stamp.sec)*1_000_000_000+int(stamp.nanosec)
    now=int(manager.get_clock().now().nanoseconds)
    return request['epoch']<value<=now


def camera_mode(node, payload):
    if ID not in payload:
        if getattr(node,'_empty_head_camera',None) is not None:node._empty_head_camera=None
        return
    key=identity(payload);mode=payload.get('event')
    now=int(node.get_clock().now().nanoseconds);epoch=payload.get(EPOCH)
    if mode=='idle':
        if payload.get('empty_head_expected_mode') not in ('markers','books'):
            raise RuntimeError('empty head camera expected mode invalid')
        node._empty_head_camera=dict(key,mode=payload['empty_head_expected_mode'],epoch=None)
    else:
        pending=getattr(node,'_empty_head_camera',None)
        if (pending is None or any(pending[k]!=v for k,v in key.items()) or mode!=pending['mode']
                or type(epoch) is not int or not 0<=now-epoch<=350_000_000
                or (pending['epoch'] is not None and epoch<pending['epoch'])):
            raise RuntimeError('empty head camera epoch invalid')
        node._empty_head_camera=dict(key,mode=mode,epoch=epoch)
    node.marker_history.clear();node.book_history.clear();node.target_tracker.reset()
    node.last_tracking_depth_ns=-1


def camera_ready(node):
    request=getattr(node,'_empty_head_camera',None)
    if request is None or node.mode not in ('markers','books'):return True
    if request['epoch'] is None or node.mode!=request['mode']:return False
    now=int(node.get_clock().now().nanoseconds)
    try:
        stamps=[int(msg.header.stamp.sec)*1_000_000_000+int(msg.header.stamp.nanosec)
            for msg in (node.latest_rgb_message,node.latest_depth_message)]
        return all(request['epoch']<stamp<=now for stamp in stamps)
    except (AttributeError,TypeError,ValueError):return False
