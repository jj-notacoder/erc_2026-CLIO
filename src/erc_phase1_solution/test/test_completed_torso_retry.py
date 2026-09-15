"""Actual hold gate/sender behavior under bounded new-feedback scheduling.

No physical hold, no motion or stable-clock guarantee is inferred from these
deterministic controller/message fixtures.
"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from erc_phase1_solution import completed_torso_hold as hold
from test_completed_torso_hold import fixture, complete


def retry_fixture(monkeypatch):
    n,sent,status,checks,clock,handle=fixture(monkeypatch)
    complete(n,clock)
    # Setup itself uses the ordinary sender and its two contact checks.
    # Assertions below cover only the subsequent action under test.
    assert checks==['contact','contact']
    checks.clear()
    n.settled_place_torso_retry_enabled=True
    wall=[10.]
    waits=[]
    monkeypatch.setattr(hold,'time',NS(monotonic=lambda:wall[0]))
    def wait(seconds):
        assert not n._lock.locked()
        assert not n._adaptive_command_guard()._is_owned()
        waits.append(seconds)
        wall[0]+=seconds
        clock[0]+=round(seconds*1e9)
        return n._cancel.is_set()
    n._cancel.wait=wait
    return n,sent,status,checks,clock,wall,waits,wait


def record(status):
    rows=[v for e,v in status if e=='torso_hold_retry_completed']
    assert len(rows)==1
    return rows[0]


@pytest.mark.parametrize('value',[True,False])
def test_strict_boolean(value):
    assert hold.checked_torso_hold_retry_enabled(value) is value


@pytest.mark.parametrize('value',[0,1,0.,1.,None,'true',[],math.nan])
def test_ambiguous_optin_rejected(value):
    with pytest.raises(ValueError,match='must be Boolean'):
        hold.checked_torso_hold_retry_enabled(value)


def test_immediate_pass_preserves_current_three_checks_and_has_no_wait(monkeypatch):
    n,sent,status,checks,clock,wall,waits,_=retry_fixture(monkeypatch)
    initial=(clock[0],wall[0]);completed=n._torso_hold_state.completed
    assert n._move_torso(.35,2.2,allow_completed_hold=True)
    assert len(sent)==1 and not waits and (clock[0],wall[0])==initial
    assert n._torso_hold_state.completed is completed
    assert checks==['contact','scene','contact']
    assert record(status)['outcome']=='admitted' and record(status)['retries']==0


def test_default_off_uses_the_original_one_shot_fallback(monkeypatch):
    n,sent,status,checks,clock,wall,waits,_=retry_fixture(monkeypatch)
    n.settled_place_torso_retry_enabled=False
    n._joint_velocities[hold.JOINT]=2e-6
    assert n._move_torso(.35,2.2,allow_completed_hold=True)
    assert len(sent)==2 and not waits
    assert not any(e=='torso_hold_retry_completed' for e,_ in status)


def test_non_place_call_never_enters_retry(monkeypatch):
    outcomes=[]
    for enabled in (False,True):
        with monkeypatch.context() as scoped:
            n,sent,status,checks,clock,wall,waits,_=retry_fixture(scoped)
            n.settled_place_torso_retry_enabled=enabled
            n._joint_velocities[hold.JOINT]=2e-6
            assert n._move_torso(.35,2.2)
            assert len(sent)==2 and not waits and checks
            assert not any(e=='torso_hold_retry_completed' for e,_ in status)
            outcomes.append((sent[-1],list(checks)))
    # Non-PLACE still sends the exact ordinary goal with its ordinary gates.
    assert outcomes[0]==outcomes[1]


@pytest.mark.parametrize('bad',['velocity','position','old_stamp','future_stamp','not_after_completion'])
def test_transient_measurement_requires_new_message_then_all_checks(monkeypatch,bad):
    n,sent,status,checks,clock,wall,waits,wait=retry_fixture(monkeypatch)
    if bad=='velocity':n._joint_velocities[hold.JOINT]=1.000001e-6
    elif bad=='position':n.joints[hold.JOINT]-=1.000001e-6
    elif bad=='old_stamp':n._joint_stamps_ns[hold.JOINT]=clock[0]-150_000_001
    elif bad=='future_stamp':n._joint_stamps_ns[hold.JOINT]=clock[0]+50_000_001
    else:n._joint_stamps_ns[hold.JOINT]=n._torso_hold_state.completed[3]
    # Corrected data must be from a genuinely newer producer. A future sample
    # therefore cannot be repaired by regressing its stamp.
    future=n._joint_stamps_ns[hold.JOINT]
    def update(seconds):
        wait(seconds)
        n.joints[hold.JOINT]=.35;n._joint_velocities[hold.JOINT]=1e-6
        clock[0]=max(clock[0],future)
        n._joint_stamps_ns[hold.JOINT]=clock[0]+1
    n._cancel.wait=update
    assert n._move_torso(.35,2.2,allow_completed_hold=True)
    assert len(sent)==1 and waits and checks==['contact','scene','contact']
    assert record(status)['retries']==1
    assert sum(record(status)['rejection_counts'].values())==1


@pytest.mark.parametrize('change',['delivery','feedback'])
def test_delivery_race_still_rejects_each_mixed_attempt(monkeypatch,change):
    n,sent,status,checks,clock,wall,waits,wait=retry_fixture(monkeypatch)
    seen=[]
    def scene(reference):
        seen.append(reference)
        if len(seen)==1:
            if change=='delivery':n._contact_generation+=1
            else:n._gripper_feedback_samples.append(object())
    def update(seconds):
        wait(seconds);n._joint_stamps_ns[hold.JOINT]=clock[0]
    n._cancel.wait=update
    assert hold.try_completed_torso_hold(n,.35,require_contact=lambda:checks.append('contact'),check_scene=scene)
    assert len(sent)==1 and len(seen)==2 and len(checks)==4 and waits
    assert record(status)['retries']==1


@pytest.mark.parametrize('bad',['missing','pending','uncertain','target','nonfinite','out_of_limits','bool_stamp'])
def test_nonretryable_input_has_no_added_wait(monkeypatch,bad):
    n,sent,status,checks,clock,wall,waits,_=retry_fixture(monkeypatch)
    target=.35
    if bad=='missing':n._torso_hold_state.completed=None
    elif bad=='pending':n._goal_handles.append(object())
    elif bad=='uncertain':n._torso_hold_state.uncertain=True
    elif bad=='target':target=.34
    elif bad=='nonfinite':n._joint_velocities[hold.JOINT]=math.nan
    elif bad=='out_of_limits':n.joints[hold.JOINT]=.36
    else:n._joint_stamps_ns[hold.JOINT]=True
    assert hold.try_completed_torso_hold(n,target,require_contact=lambda:None,check_scene=lambda r:None) is None
    assert not waits and len(sent)==1 and record(status)['retries']==0


@pytest.mark.parametrize('change',['cancel','cancel_clear','generation','completed','pending','uncertain','scene','held_identity','held_values','epoch','hazard','sensor_fault','raw_fault','robot_contact'])
def test_retry_invalidation_exits_without_action_or_second_wait(monkeypatch,change):
    n,sent,status,checks,clock,wall,waits,wait=retry_fixture(monkeypatch)
    n._joint_velocities[hold.JOINT]=2e-6
    def invalidate(seconds):
        wait(seconds)
        if change=='cancel':n._cancel.set()
        elif change=='cancel_clear':n._cancel.set();n._cancel.clear()
        elif change=='generation':n._torso_hold_state.generation+=1
        elif change=='completed':n._torso_hold_state.completed=tuple(list(n._torso_hold_state.completed))
        elif change=='pending':n._pending_retained_acceptances.add(object())
        elif change=='uncertain':n._torso_hold_state.uncertain=True
        elif change=='scene':n._active_place_scene_reference={}
        elif change=='held_identity':n._held_book_corners=n._held_book_corners.copy()
        elif change=='held_values':n._held_book_corners[0,0]+=.001
        elif change=='epoch':n._contact_epoch+=1
        elif change=='hazard':n._payload_hazard_latched='lost'
        elif change=='sensor_fault':n._held_grip_sensor_fault='bad'
        elif change=='raw_fault':n._raw_contacts_first_failure='bad'
        else:n._target_robot_contact_latched=True
    n._cancel.wait=invalidate
    assert n._move_torso(.35,2.2,allow_completed_hold=True) is False
    assert len(sent)==1 and len(waits)==1 and record(status)['outcome']=='failed'


@pytest.mark.parametrize('fault',['contact','scene','payload'])
def test_actual_gate_fault_does_not_wait_or_send_fallback(monkeypatch,fault):
    n,sent,status,checks,clock,wall,waits,_=retry_fixture(monkeypatch)
    if fault=='payload':n._payload_hazard_reason=lambda **kw:'lost'
    def call(kind):
        if fault==kind:raise RuntimeError(kind)
    if fault=='payload':
        assert hold.try_completed_torso_hold(n,.35,require_contact=lambda:None,check_scene=lambda r:None) is False
    else:
        with pytest.raises(RuntimeError,match=fault):
            hold.try_completed_torso_hold(n,.35,require_contact=lambda:call('contact'),check_scene=lambda r:call('scene'))
    assert not waits and len(sent)==1 and record(status)['outcome']=='failed'


@pytest.mark.parametrize('mode',['wall_frozen_ros','ros_first','ten_retries','delivery_churn'])
def test_caps_fall_back_once_with_exact_original_goal(monkeypatch,mode):
    n,sent,status,checks,clock,wall,waits,wait=retry_fixture(monkeypatch)
    n._joint_velocities[hold.JOINT]=2e-6
    start_ros=clock[0]
    def advance(seconds):
        wait(seconds)
        if mode=='wall_frozen_ros':clock[0]=start_ros
        elif mode=='ros_first':clock[0]=start_ros+200_000_000
        else:n._joint_stamps_ns[hold.JOINT]=clock[0]
    n._cancel.wait=advance
    if mode=='delivery_churn':
        n._joint_velocities[hold.JOINT]=0.
        from erc_phase1_solution import manipulation_node as runtime
        monkeypatch.setattr(runtime,'measured_scene_context',lambda n,r:setattr(n,'_contact_generation',n._contact_generation+1))
    assert n._move_torso(.35,2.2,allow_completed_hold=True)
    assert len(sent)==2
    point=sent[-1].trajectory.points[0]
    assert list(point.positions)==[.35] and point.time_from_start.sec==2 and point.time_from_start.nanosec==200_000_000
    assert not point.velocities and not point.accelerations
    result=record(status)
    expected={'wall_frozen_ros':'wall_limit','ros_first':'ros_limit','ten_retries':'retry_limit','delivery_churn':'retry_limit'}[mode]
    assert result['reason']==expected and result['outcome']=='fallback_required'
    assert result['retries']<=10 and result['wall_seconds']<=1.0000000001


@pytest.mark.parametrize('which',['ros','wall','producer'])
def test_clock_or_producer_regression_fails_without_goal(monkeypatch,which):
    n,sent,status,checks,clock,wall,waits,wait=retry_fixture(monkeypatch)
    n._joint_velocities[hold.JOINT]=2e-6
    def regress(seconds):
        wait(seconds)
        if which=='ros':clock[0]-=50_000_000
        elif which=='wall':wall[0]-=.05
        else:n._joint_stamps_ns[hold.JOINT]-=1
    n._cancel.wait=regress
    assert n._move_torso(.35,2.2,allow_completed_hold=True) is False
    assert len(sent)==1 and len(waits)==1
    assert record(status)['reason'] in ('clock_regressed','measurement_regressed')


@pytest.mark.parametrize('when',['first','second','before_completed'])
def test_clock_regression_inside_gate_is_fatal_only_when_retry_enabled(monkeypatch,when):
    n,sent,status,checks,clock,wall,waits,_=retry_fixture(monkeypatch)
    if when=='first':
        calls=[];original=clock[0]
        def now():
            calls.append(None)
            return NS(nanoseconds=original if len(calls)==1 else original-1)
        n.get_clock=lambda:NS(now=now)
    elif when=='before_completed':
        clock[0]=n._torso_hold_state.completed[3]-1
    else:
        from erc_phase1_solution import manipulation_node as runtime
        monkeypatch.setattr(runtime,'measured_scene_context',lambda n,r:clock.__setitem__(0,clock[0]-1))
    assert n._move_torso(.35,2.2,allow_completed_hold=True) is False
    assert len(sent)==1 and not waits and record(status)['reason']=='clock_regressed'


@pytest.mark.parametrize('regression_ns',[1,150_000_001,2_000_000_000])
def test_producer_regression_in_second_gate_cannot_send_fallback(monkeypatch,regression_ns):
    n,sent,status,checks,clock,wall,waits,_=retry_fixture(monkeypatch)
    from erc_phase1_solution import manipulation_node as runtime
    monkeypatch.setattr(runtime,'measured_scene_context',
        lambda n,r:n._joint_stamps_ns.__setitem__(hold.JOINT,n._joint_stamps_ns[hold.JOINT]-regression_ns))
    assert n._move_torso(.35,2.2,allow_completed_hold=True) is False
    assert len(sent)==1 and not waits and record(status)['reason']=='measurement_regressed'


@pytest.mark.parametrize('change',['base','right','head','unlatched_hazard'])
def test_post_wait_fallback_checks_actual_scene_and_live_payload(monkeypatch,change):
    n,sent,status,checks,clock,wall,waits,wait=retry_fixture(monkeypatch)
    from erc_phase1_solution import manipulation_node as runtime
    from erc_phase1_solution.placement_scene_context import measured_scene_context
    n.right_chain=NS(active_names=['torso_lift_joint','arm_right_1_joint'])
    n.head_chain=NS(active_names=['head_1_joint','head_2_joint'])
    parked=dict(arm_right_1_joint=0.,head_1_joint=0.,head_2_joint=0.)
    n.joints.update(parked)
    n._joint_stamps_ns.update({k:clock[0] for k in parked})
    n._staging_odom=dict(stamp_ns=clock[0],pose=[0.,0.,0.],linear_speed=0.,angular_speed=0.)
    n._active_place_scene_reference=dict(base_pose=[0.,0.,0.],parked_joints=parked.copy())
    monkeypatch.setattr(runtime,'measured_scene_context',measured_scene_context)
    n._joint_velocities[hold.JOINT]=2e-6
    def change_at_expiry(seconds):
        wait(seconds)
        clock[0]+=200_000_000  # Expire before a fresh measurement retries.
        n._staging_odom['stamp_ns']=clock[0]
        n._joint_stamps_ns.update({k:clock[0] for k in parked})
        if change=='base':n._staging_odom['pose'][0]=.002001
        elif change=='right':n.joints['arm_right_1_joint']=.001001
        elif change=='head':n.joints['head_2_joint']=.001001
        else:n._payload_hazard_reason=lambda **kw:'contact_lost'
    n._cancel.wait=change_at_expiry
    if change=='unlatched_hazard':
        assert n._move_torso(.35,2.2,allow_completed_hold=True) is False
        assert record(status)['reason']=='fallback_payload_hazard'
    else:
        with pytest.raises(RuntimeError,match='placement_scene_'):
            n._move_torso(.35,2.2,allow_completed_hold=True)
    assert len(sent)==1 and len(waits)==1 and record(status)['outcome']=='failed'


def test_fallback_context_change_during_admission_is_fatal(monkeypatch):
    n,sent,status,checks,clock,wall,waits,wait=retry_fixture(monkeypatch)
    n._joint_velocities[hold.JOINT]=2e-6
    def expire(seconds):wait(seconds);clock[0]+=200_000_000
    n._cancel.wait=expire
    from erc_phase1_solution import manipulation_node as runtime
    monkeypatch.setattr(runtime,'measured_scene_context',lambda n,r:setattr(n,'_contact_epoch',n._contact_epoch+1))
    assert n._move_torso(.35,2.2,allow_completed_hold=True) is False
    assert len(sent)==1 and record(status)['reason']=='held_context_changed'


@pytest.mark.parametrize('counter',['producer','ros','wall'])
def test_regression_during_fallback_admission_cannot_send_goal(monkeypatch,counter):
    n,sent,status,checks,clock,wall,waits,wait=retry_fixture(monkeypatch)
    n._joint_velocities[hold.JOINT]=2e-6
    def expire(seconds):wait(seconds);clock[0]+=200_000_000
    n._cancel.wait=expire
    from erc_phase1_solution import manipulation_node as runtime
    def scene(node,reference):
        if counter=='producer':node._joint_stamps_ns[hold.JOINT]-=1
        elif counter=='ros':clock[0]-=1
        else:wall[0]-=.0001
    monkeypatch.setattr(runtime,'measured_scene_context',scene)
    assert n._move_torso(.35,2.2,allow_completed_hold=True) is False
    assert len(sent)==1 and len(waits)==1
    assert record(status)['reason'] in ('clock_regressed','measurement_regressed')


def test_deadline_during_retried_gate_cannot_publish_skip(monkeypatch):
    n,sent,status,checks,clock,wall,waits,wait=retry_fixture(monkeypatch)
    n._joint_velocities[hold.JOINT]=2e-6
    def ready(seconds):
        wait(seconds);n._joint_stamps_ns[hold.JOINT]=clock[0];n._joint_velocities[hold.JOINT]=0.
    n._cancel.wait=ready
    def scene(reference):wall[0]=11.
    assert hold.try_completed_torso_hold(n,.35,require_contact=lambda:None,check_scene=scene) is None
    assert not any(e=='torso_motion_skipped' for e,_ in status)
    assert record(status)['reason']=='wall_limit'


def test_no_new_input_never_retries_the_same_measurement(monkeypatch):
    n,sent,status,checks,clock,wall,waits,_=retry_fixture(monkeypatch)
    n._joint_velocities[hold.JOINT]=2e-6
    assert n._move_torso(.35,2.2,allow_completed_hold=True)
    assert record(status)['retries']==0 and record(status)['reason']=='ros_limit'
    assert len(sent)==2


def test_original_measurement_boolean_and_limits_remain_exact():
    # A mandatory original-source pin is supplied by the preparation runner.
    import os
    baseline=Path(os.environ['TORSO_RETRY_BASE_SOURCE'])/'completed_torso_hold.py'
    original=ast.parse(baseline.read_text())
    current=ast.parse(Path(hold.__file__).read_text())
    def node(tree,name):return next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    old=node(original,'_measurement_locked');new=node(current,'_measurement_locked')
    old_if=next(x for x in old.body if isinstance(x,ast.If))
    assert ast.dump(old_if.test) in [ast.dump(x.test) for x in new.body if isinstance(x,ast.If)]
    assert ast.dump(old.body[0].body[0])==ast.dump(new.body[0].body[0])
    old_gate=node(original,'try_completed_torso_hold')
    new_gate=node(current,'_try_completed_torso_hold_once')
    # Every original condition, including both mixed-delivery rejections and
    # action/held/scene identities, remains present verbatim in the gate AST.
    old_conditions=[ast.dump(x.test) for x in ast.walk(old_gate) if isinstance(x,ast.If)]
    new_conditions=[ast.dump(x.test) for x in ast.walk(new_gate) if isinstance(x,ast.If)]
    for condition in old_conditions:assert condition in new_conditions
