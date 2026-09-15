"""Actual PLACE suffix/command paths and owned raw-feedback completion gates."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import threading

import pytest

from erc_phase1_solution import release_pose_finish as finish
from erc_phase1_solution import empty_head_timing as actual_empty_head_timing
from erc_phase1_solution import place_contact_guard as contacts
from erc_phase1_solution.motion_profiles import IK_JOINTS
from erc_phase1_solution.release_evidence import AttemptIdentity


IDENTITY = AttemptIdentity('trial', 'attempt', 'book_col_3_row_2_red')
BOOK = IDENTITY.target_model+'::book_link::collision'
BIN = 'erc_collection_bin::bin_link::collision'
ROBOT = 'tiago_pro::gripper_left_fingertip_left_link::collision'
GOAL = (.3, .4, .2, .1, .1, .1, .1, .1)


def actual_methods():
    path = Path(__file__).resolve().parents[1]/'erc_phase1_solution/manipulation_node.py'
    cls = next(x for x in ast.parse(path.read_text()).body
               if isinstance(x, ast.ClassDef) and x.name == 'ManipulationNode')
    methods = {x.name: x for x in cls.body if isinstance(x, ast.FunctionDef)}
    start = next(i for i, x in enumerate(methods['_place'].body)
                 if isinstance(x, ast.Assign) and isinstance(x.targets[0], ast.Name)
                 and x.targets[0].id == 'release_owner')
    signature = ('self, correlation, solutions, direct_empty_home, '
                 'carried_transition_waypoints, torso_ready, '
                 'unloaded_home_waypoints, arm_timing_options')
    suffix = ast.parse('def suffix('+signature+'):\n    pass').body[0]
    initialization = methods['_place'].body[0]
    assert isinstance(initialization, ast.Assign)
    assert [target.id for target in initialization.targets] == ['release_only_request', 'release_only_endpoint']
    assert isinstance(initialization.value, ast.Constant) and initialization.value.value is None
    suffix.body = [initialization, *methods['_place'].body[start:]]
    body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
            suffix, methods['_run_command'], methods['_on_command'],
            methods['_hold_adaptive_gripper'], methods['_open_gripper'],
            methods['_clear_target_contact_samples_unlocked']]
    scope = dict(__package__='erc_phase1_solution', _empty_head_timing=actual_empty_head_timing, release_pose_finish=finish,
                 _deactivate_place_contacts=contacts.deactivate,
                 _place_contact_fault=contacts.fault_reason,
                 measured_scene_context=finish.measured_scene_context,
                 decode_event=lambda value:value)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 str(path), 'exec'), scope)
    return scope


@pytest.mark.parametrize('enabled', [False, True])
def test_actual_normal_place_suffix_keeps_opening_and_selects_honest_completion(enabled):
    scope = actual_methods(); calls = []; owner = object()
    n = NS(place_finish_at_release_enabled=enabled)
    def opened(*, verify_measurement):
        calls.append('open'); return verify_measurement()
    n._open_gripper = opened
    n._return_from_bin = lambda *a, **k: calls.append('return') or True
    n._recover_closed_place = lambda **k: pytest.fail('normal release attempted recovery')
    def measured(node, correlation, goal, event):
        calls.append(event); return dict(verified=True)
    scope['observe_measured_open_pose'] = measured
    scope['raised_place_normal_finish'] = lambda *a: (calls.append('return_options') or {}, GOAL)
    scope['HOME'] = GOAL
    scope['release_pose_finish'] = NS(
        prepare=lambda *a: calls.append('prepare') or owner,
        remember_open=lambda *a: calls.append('remember_open') or True,
        finish=lambda *a: calls.append('finish_release_pose') or True)
    assert scope['suffix'](n, vars(IDENTITY), [GOAL], [GOAL], [], GOAL, [], {})
    expected = (['prepare', 'open', 'placement_open_measured', 'remember_open', 'finish_release_pose']
                if enabled else ['open', 'placement_open_measured', 'return_options',
                                 'return', 'placement_hand_return_measured'])
    assert calls == expected


@pytest.mark.parametrize('failure', ['opening', 'hold'])
def test_actual_selected_suffix_failure_never_reopens_or_returns(failure):
    scope = actual_methods(); calls = []; n = NS(place_finish_at_release_enabled=True)
    n._open_gripper = lambda **k: calls.append('open') or (failure != 'opening' and k['verify_measurement']())
    n._return_from_bin = lambda *a, **k: pytest.fail('post-release return on fault')
    n._recover_closed_place = lambda **k: pytest.fail('recovery on uncertain release')
    scope['observe_measured_open_pose'] = lambda *a: dict(verified=True)
    def reject(*a): raise RuntimeError('hold_fault')
    scope['release_pose_finish'] = NS(prepare=lambda *a:object(),
        remember_open=lambda *a:True, finish=reject)
    with pytest.raises(RuntimeError, match='placement_release_unverified|hold_fault'):
        scope['suffix'](n, vars(IDENTITY), [GOAL], [GOAL], [], GOAL, [], {})
    assert calls == ['open']


def fixture(monkeypatch, *, opened=True):
    monkeypatch.setattr(finish.time, 'monotonic', lambda: 1.)
    n = NS(_lock=threading.Lock(), _command_lock=threading.RLock(),
        _cancel=threading.Event(), _busy=True, _goal_handles=[],
        _pending_retained_acceptances=[], place_finish_at_release_enabled=True,
        delivery_evidence_enabled=True, table_scene_required=True, bin_scene_required=True,
        gripper_open=.069, _gripper_open_confirmed=False, _release_pose_owner=None,
        _release_pose_fault_latched=None, _held_book_corners=object(),
        _target_book_model=IDENTITY.target_model, chain=object(),
        right_chain=NS(active_names=('arm_right_1_joint',)),
        head_chain=NS(active_names=('head_1_joint',)), now=1_000_000_000,
        events=[], commands=[], errors=[])
    n._adaptive_command_guard = lambda:n._command_lock
    n.get_clock = lambda:NS(now=lambda:NS(nanoseconds=n.now))
    n._publish_status = lambda event, **fields:n.events.append((event, fields))
    n.get_logger = lambda:NS(error=n.errors.append)
    n._active_place_scene_reference = dict(base_pose=[0.,0.,0.],
        parked_joints=dict(arm_right_1_joint=0., head_1_joint=0.))
    n._place_contact_guard = contacts.PlaceContactGuard(n.now, vars(IDENTITY).copy())
    def refresh(stamp):
        n.now=stamp
        n.joints=dict(zip(IK_JOINTS, GOAL), arm_right_1_joint=0., head_1_joint=0.,
                      gripper_left_finger_joint=.069)
        n._joint_stamps_ns={name:stamp for name in n.joints}
        n._staging_odom=dict(stamp_ns=stamp,pose=(0.,0.,0.),linear_speed=0.,angular_speed=0.)
    n.refresh=refresh; refresh(n.now)
    owner=finish.prepare(n, vars(IDENTITY), GOAL, [])
    if not opened:return n,owner
    record_feedback(n, n.now)
    assert finish.remember_open(n,owner,dict(vars(IDENTITY),verified=True,
        producer_stamp_ns=n.now,raw_sample_sequence=n._delivery_sample_sequence))
    n._held_book_corners=n._target_book_model=None; n._gripper_open_confirmed=True
    return n,owner


def contact_message(stamp, pairs):
    return NS(header=NS(stamp=NS(sec=stamp//10**9,nanosec=stamp%10**9)),
        contacts=[NS(collision1=NS(name=a),collision2=NS(name=b)) for a,b in pairs])


def ready(n,owner):
    for stamp in range(n.now+50_000_000,n.now+600_000_001,50_000_000):
        n.refresh(stamp)
        record_feedback(n, stamp)
        finish.observe_contacts(n,contact_message(stamp,[(BOOK,BIN)]),'/bin_contacts')
    assert finish.finish(n,owner)


def test_owned_actual_raw_callbacks_and_final_locked_measurement(monkeypatch):
    n,owner=fixture(monkeypatch);ready(n,owner)
    assert owner.phase=='ready' and not n.events
    assert finish.publish_terminal(n,vars(IDENTITY))
    assert [event for event,_ in n.events]==['placement_release_pose_measured','succeeded']
    measured=n.events[0][1]
    assert measured['hand_return'] is measured['hand_clear_verified'] is False
    assert measured['physical_inside_verified'] is None
    assert measured['received_contact_fault_latch_clear'] is True
    assert measured['contact_stream_coverage_verified'] is False
    assert 'no_observed_target_robot_contact_verified' not in measured
    assert n.events[1][1]['completion_mode']=='release_pose'
    assert n.events[1][1]['hand_return'] is False
    finish.deactivate(n)
    assert n._release_pose_owner is None and n._release_pose_fault_latched is None
    assert not n.commands


@pytest.mark.parametrize('fault', ['cancel','goal','pending','target','held','scene_object',
    'scene_value','guard_object','guard_fault','chain','parked_joint','stale','clock_reverse'])
def test_context_or_feedback_fault_after_ready_cannot_publish_success(monkeypatch,fault):
    n,owner=fixture(monkeypatch);ready(n,owner)
    if fault=='cancel':n._cancel.set()
    if fault=='goal':n._goal_handles.append(object())
    if fault=='pending':n._pending_retained_acceptances.append(object())
    if fault=='target':n._target_book_model=IDENTITY.target_model
    if fault=='held':n._held_book_corners=object()
    if fault=='scene_object':n._active_place_scene_reference=dict(n._active_place_scene_reference)
    if fault=='scene_value':n._active_place_scene_reference['base_pose'][0]=.01
    if fault=='guard_object':n._place_contact_guard=contacts.PlaceContactGuard(n.now,vars(IDENTITY))
    if fault=='guard_fault':n._place_contact_guard.fault={'reason':'actual_contact'}
    if fault=='chain':n.chain=object()
    if fault=='parked_joint':n.joints['head_1_joint']=.002
    if fault=='stale':n.now+=160_000_000
    if fault=='clock_reverse':n.now-=1
    with pytest.raises(RuntimeError):finish.publish_terminal(n,vars(IDENTITY))
    finish.deactivate(n)
    assert n._cancel.is_set() and n._release_pose_fault_latched
    assert not n.events and not n.commands


@pytest.mark.parametrize('pair', [(BOOK,ROBOT),(ROBOT,BIN)])
def test_actual_contact_guard_latches_postopen_without_gripper_command(monkeypatch,pair):
    n,owner=fixture(monkeypatch);ready(n,owner)
    n._hold_adaptive_gripper=lambda *a:pytest.fail('post-open hold command forbidden')
    n.now+=50_000_000
    contacts.observe(n,contact_message(n.now,[pair]))
    assert n._cancel.is_set()
    with pytest.raises(RuntimeError):finish.publish_terminal(n,vars(IDENTITY))
    assert not any(event=='succeeded' for event,_ in n.events)
    finish.deactivate(n);assert n._release_pose_fault_latched


def test_raw_scene_contact_latches_before_legacy_diagnostic_can_race_terminal(monkeypatch):
    n,owner=fixture(monkeypatch);ready(n,owner);n.now+=50_000_000
    finish.observe_contacts(n,contact_message(n.now,[(ROBOT,BIN)]),'/contacts')
    assert n._place_contact_guard.fault is None
    assert owner.evidence.fault=='release_pose_scene_contact'
    with pytest.raises(RuntimeError):finish.publish_terminal(n,vars(IDENTITY))
    assert not n.events


@pytest.mark.parametrize('pair', [(BOOK,ROBOT),(ROBOT,BIN)])
def test_contact_between_measured_open_stamp_and_owner_handoff_is_not_lost(monkeypatch,pair):
    n,owner=fixture(monkeypatch,opened=False)
    record_feedback(n,1_050_000_000);n.refresh(1_100_000_000)
    finish.observe_contacts(n,contact_message(1_075_000_000,[pair]),'/contacts')
    assert len(owner.opening_contacts)==1
    assert not finish.remember_open(n,owner,dict(vars(IDENTITY),verified=True,
        producer_stamp_ns=1_050_000_000,raw_sample_sequence=1))
    assert n._cancel.is_set() and owner.evidence.fault


def test_actual_raw_adapter_remains_active_after_open_measurement_stops(monkeypatch):
    from erc_phase1_solution import release_sensor_adapter as adapter
    n,owner=fixture(monkeypatch);n._delivery_measurement_active=False
    n.refresh(1_050_000_000)
    header=NS(stamp=NS(sec=1,nanosec=50_000_000))
    adapter.record_raw_odom(n,NS(header=header,twist=NS(twist=NS(
        linear=NS(x=0.,y=0.),angular=NS(z=0.)))))
    names=list(n.joints)
    adapter.record_raw_joints(n,NS(header=header,name=names,
        position=[n.joints[name] for name in names],velocity=[0.]*len(names)))
    assert owner.evidence.last_joint_stamp==n.now
    assert owner.evidence.fault is None
    # Actual short velocity array becomes missing/NaN, never fabricated zero.
    header.stamp.nanosec=100_000_000;n.now=1_100_000_000
    adapter.record_raw_joints(n,NS(header=header,name=names,
        position=[n.joints[name] for name in names],velocity=[]))
    assert owner.evidence.fault and n._cancel.is_set()


@pytest.mark.parametrize('fault',['drift','velocity','missing_velocity'])
def test_transient_parked_geometry_fault_latches_before_next_poll(monkeypatch,fault):
    n,owner=fixture(monkeypatch);ready(n,owner);n.refresh(n.now+50_000_000)
    raw=dict(producer_stamp_ns=n.now,sequence=n._delivery_sample_sequence+1,positions=dict(n.joints),
        velocities={name:0. for name in n.joints},odom=dict(n._staging_odom))
    if fault=='drift':raw['positions']['head_1_joint']=.002
    if fault=='velocity':raw['velocities']['arm_right_1_joint']=.002
    if fault=='missing_velocity':del raw['velocities']['head_1_joint']
    with n._lock:finish.observe_joint_locked(n,raw)
    assert owner.evidence.fault=='release_pose_parked_feedback_changed'
    with pytest.raises(RuntimeError):finish.publish_terminal(n,vars(IDENTITY))
    assert not n.events


def command_fixture(n):
    n.dry_run=False;n.book_overview_tilt=0.;n.book_row_tilts=[]
    def forbidden(*a,**k):pytest.fail('post-open actuator or planner invoked')
    n._stow=n._move_head=n._pick=n._compact_transport=n._open_gripper=forbidden
    n._place=lambda payload:True


def test_actual_run_command_owns_until_distinct_completion_then_cleans_up(monkeypatch):
    n,owner=fixture(monkeypatch);ready(n,owner);command_fixture(n)
    actual_methods()['_run_command'](n,'place',dict(vars(IDENTITY),completion_mode='release_pose',release_evidence_contract=finish.CONTRACT))
    assert [event for event,_ in n.events]==['started','placement_release_pose_measured','succeeded']
    assert not n._busy and n._place_contact_guard is None and n._release_pose_owner is None
    assert n._active_place_scene_reference is None and n._release_pose_fault_latched is None


def test_actual_run_command_cannot_claim_success_if_final_feedback_changed(monkeypatch):
    n,owner=fixture(monkeypatch);ready(n,owner);command_fixture(n)
    n._place=lambda payload:setattr(n,'now',n.now+160_000_000) or True
    actual_methods()['_run_command'](n,'place',dict(vars(IDENTITY),completion_mode='release_pose',release_evidence_contract=finish.CONTRACT))
    assert [event for event,_ in n.events]==['started','failed']
    assert n._release_pose_fault_latched and n._cancel.is_set() and not n._busy


@pytest.mark.parametrize('active', [True,False])
def test_actual_new_command_is_rejected_by_live_owner_or_fault_latch(monkeypatch,active):
    n,owner=fixture(monkeypatch);ready(n,owner)
    if not active:finish.deactivate(n)
    actual_methods()['_on_command'](n,NS(data={'event':'stow'}))
    assert n.events[-1][0]=='rejected'
    assert n.events[-1][1]['reason']=='release_pose_owner_or_fault_active'


def test_actual_cancel_after_open_cannot_issue_recovery_or_gripper_hold(monkeypatch):
    n,owner=fixture(monkeypatch);ready(n,owner)
    actual_methods()['_on_command'](n,NS(data={'event':'cancel'}))
    assert n._cancel.is_set() and owner.evidence.fault=='release_pose_cancelled'
    assert n.events[-1][0]=='cancelled'
    with pytest.raises(RuntimeError):finish.finish(n,owner)


@pytest.mark.parametrize('enabled,requested', [(False,'release_pose'),(True,None),(True,'hand_return')])
def test_local_completion_mode_cannot_be_changed_by_payload(enabled,requested):
    with pytest.raises(RuntimeError,match='mode_mismatch'):
        finish.validate_mode(NS(place_finish_at_release_enabled=enabled),{'completion_mode':requested})


def test_disabled_preparation_has_no_owner_or_sensor_requirements():
    assert finish.prepare(NS(place_finish_at_release_enabled=False),None,None,None) is None


def actual_mission_methods():
    from erc_phase1_solution.release_evidence import ReleaseEvidence
    from erc_phase1_solution.release_pose_evidence import ReleasePoseEvidence,MODE,EVENT,CONTRACT
    path=Path(__file__).resolve().parents[1]/'erc_phase1_solution/mission_manager.py'
    cls=next(x for x in ast.parse(path.read_text()).body
             if isinstance(x,ast.ClassDef) and x.name=='MissionManager')
    selected=[x for x in cls.body if isinstance(x,ast.FunctionDef)
              and x.name in ('_manipulate','_on_manipulation_status')]
    assert len(selected)==2
    scope=dict(__package__='erc_phase1_solution',_empty_head_timing=actual_empty_head_timing,AttemptIdentity=AttemptIdentity,
        ReleaseEvidence=ReleaseEvidence,ReleasePoseEvidence=ReleasePoseEvidence,
        RELEASE_POSE_MODE=MODE,RELEASE_POSE_EVENT=EVENT,RELEASE_EVIDENCE_CONTRACT=CONTRACT,decode_event=lambda value:value,
        uuid=NS(uuid4=lambda:NS(hex='attempt')),time=NS(monotonic=lambda:1.))
    module=ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')],level=0),*selected],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),scope)
    return scope


@pytest.mark.parametrize('enabled',[False,True])
def test_actual_mission_binds_local_attempt_mode_and_rejects_opposite_milestone(enabled):
    from erc_phase1_solution.release_pose_evidence import ReleasePoseEvidence
    scope=actual_mission_methods();sent=[];logs=[]
    n=NS(place_finish_at_release_enabled=enabled,delivery_evidence_enabled=True,
        trial_id=IDENTITY.trial_id,target_book_model=IDENTITY.target_model,
        manip_command_pub=object(),now=1_000_000_000)
    n._command=lambda *args,**fields:sent.append((args,fields))
    n._log=lambda *args,**fields:logs.append((args,fields))
    n.get_clock=lambda:NS(now=lambda:NS(nanoseconds=n.now))
    scope['_manipulate'](n,'place')
    assert isinstance(n.delivery_release_evidence,ReleasePoseEvidence)==enabled
    assert sent[0][1].get('completion_mode')==('release_pose' if enabled else None)
    assert sent[0][1]['placement_attempt_id']=='attempt'
    fields=dict(vars(IDENTITY),event='placement_open_measured',command='place',
        producer_stamp_ns=n.now,verified=True)
    scope['_on_manipulation_status'](n,NS(data=fields))
    assert n.delivery_release_evidence.open_epoch==n.now
    # Neither an ordinary hand-return event nor a release-pose event may
    # silently switch this already chosen attempt policy.
    n.now=1_200_000_000
    fields.update(event=('placement_hand_return_measured' if enabled else
                         'placement_release_pose_measured'),producer_stamp_ns=n.now)
    scope['_on_manipulation_status'](n,NS(data=fields))
    assert n.delivery_release_evidence.fault
    assert not n._delivery_release_snapshot['operation_completed']


@pytest.mark.parametrize('active',[True,False])
def test_actual_watchdog_hold_sender_is_inert_after_measured_open_or_fault(monkeypatch,active):
    n,owner=fixture(monkeypatch);ready(n,owner)
    if not active:finish.deactivate(n)
    n._adaptive_motion_feedback=lambda:pytest.fail('post-open sender read feedback')
    n.gripper_pub=NS(publish=lambda message:pytest.fail('post-open hold published'))
    actual_methods()['_hold_adaptive_gripper'](n,'watchdog_after_failed_open')
    assert n._cancel.is_set()


def test_default_hold_sender_keeps_existing_one_shot_path():
    n=NS(_adaptive_hold_sent=True,_adaptive_command_guard=lambda:threading.RLock())
    actual_methods()['_hold_adaptive_gripper'](n,'original')
    assert n._adaptive_motion_halt_reason=='original'


def test_actual_failed_open_monitor_restoration_cannot_publish_a_postopen_hold(monkeypatch):
    n,owner=fixture(monkeypatch,opened=False)
    record_feedback(n,1_050_000_000);n.refresh(1_100_000_000)
    n._payload_monitor_enabled=True;n._retention_probe_active=False
    n._command_gripper=lambda *a,**k:True
    finish.observe_contacts(n,contact_message(1_075_000_000,[(BOOK,ROBOT)]),'/contacts')
    measured=dict(vars(IDENTITY),verified=True,producer_stamp_ns=1_050_000_000,raw_sample_sequence=1)
    actual=actual_methods()
    assert not actual['_open_gripper'](n,verify_measurement=lambda:finish.remember_open(n,owner,measured))
    assert n._payload_monitor_enabled and n._held_book_corners is not None
    n.gripper_pub=NS(publish=lambda message:pytest.fail('restored monitor moved hand'))
    n._adaptive_motion_feedback=lambda:pytest.fail('post-open sender read feedback')
    actual['_hold_adaptive_gripper'](n,'restored_monitor_contact_lost')
    assert n._cancel.is_set()



def record_feedback(n, stamp, *, bad_velocity=False):
    from erc_phase1_solution import release_sensor_adapter as adapter
    n.refresh(stamp)
    header=NS(stamp=NS(sec=stamp//10**9,nanosec=stamp%10**9))
    adapter.record_raw_odom(n,NS(header=header,twist=NS(twist=NS(
        linear=NS(x=0.,y=0.),angular=NS(z=0.)))))
    names=list(n.joints);velocities=[0.]*len(names)
    if bad_velocity:velocities[names.index('head_1_joint')]=.02
    adapter.record_raw_joints(n,NS(header=header,name=names,
        position=[n.joints[name] for name in names],velocity=velocities))


def real_open_measurement(n, monkeypatch, *, publication_callback=None):
    """Actual measurement loop, driven by deterministic raw adapter callbacks."""
    from erc_phase1_solution import release_sensor_adapter as adapter
    initial=n.now;delivered=[]
    def feed(_):
        assert not delivered, 'measurement unexpectedly requested more fixture feedback'
        delivered.append(True)
        for offset in (50_000_000,100_000_000,150_000_000):
            record_feedback(n,initial+offset)
    monkeypatch.setattr(adapter.time,'sleep',feed)
    publish=n._publish_status
    def status(event,**fields):
        publish(event,**fields)
        if event=='placement_open_measured' and fields['verified'] and publication_callback:
            publication_callback(fields)
    n._publish_status=status
    fields=adapter.observe_measured_open_pose(n,vars(IDENTITY),GOAL,'placement_open_measured')
    assert fields['verified'] and fields['raw_sample_sequence']==3
    assert not n._delivery_measurement_active
    return fields


@pytest.mark.parametrize('discarded',['opening','bad_postopen'])
def test_actual_adapter_handoff_rejects_evicted_boundary_or_transient(monkeypatch,discarded):
    n,owner=fixture(monkeypatch,opened=False)
    def overflow(fields):
        epoch=fields['producer_stamp_ns']
        # Even within freshness bounds, the selected ring cannot silently
        # discard the verified boundary or a later transient during publish.
        for index in range(1,131):
            record_feedback(n,epoch+index*1_000_000,
                bad_velocity=(discarded=='bad_postopen' and index==1))
        assert len(n._delivery_raw_samples)==128
        assert n._delivery_raw_samples[0]['sequence']>fields['raw_sample_sequence']
    fields=real_open_measurement(n,monkeypatch,publication_callback=overflow)
    assert not finish.remember_open(n,owner,fields)
    assert owner.evidence.fault=='release_pose_joint_handoff_gap'
    assert n._cancel.is_set()
    with pytest.raises(RuntimeError):finish.publish_terminal(n,vars(IDENTITY))
    assert not any(event=='succeeded' for event,_ in n.events)


@pytest.mark.parametrize('damage',['missing_field','wrong_stamp','interior_gap','sequence_reset'])
def test_open_handoff_requires_exact_boundary_and_contiguous_callback_sequence(monkeypatch,damage):
    n,owner=fixture(monkeypatch,opened=False)
    fields=real_open_measurement(n,monkeypatch)
    record_feedback(n,n.now+25_000_000)
    record_feedback(n,n.now+25_000_000)
    if damage=='missing_field':fields.pop('raw_sample_sequence')
    if damage=='wrong_stamp':fields['producer_stamp_ns']-=1
    if damage=='interior_gap':del n._delivery_raw_samples[-2]
    if damage=='sequence_reset':n._delivery_sample_sequence=1
    assert not finish.remember_open(n,owner,fields)
    assert owner.evidence.fault=='release_pose_joint_handoff_gap'
    assert n._cancel.is_set()


def test_retained_boundary_allows_only_preopening_eviction(monkeypatch):
    n,owner=fixture(monkeypatch,opened=False)
    fields=real_open_measurement(n,monkeypatch)
    for index in range(1,128):record_feedback(n,fields['producer_stamp_ns']+index*1_000_000)
    assert n._delivery_raw_samples[0]['sequence']==fields['raw_sample_sequence']
    assert finish.remember_open(n,owner,fields)
    assert owner.last_joint_sequence==n._delivery_sample_sequence
    assert owner.evidence.fault is None and not n._cancel.is_set()


@pytest.mark.parametrize('damage',['sequence_jump','first_gap'])
def test_post_handoff_first_sample_must_continue_sequence_and_producer_epoch(monkeypatch,damage):
    n,owner=fixture(monkeypatch)
    if damage=='sequence_jump':n._delivery_sample_sequence+=1
    record_feedback(n,n.now+(76_000_000 if damage=='first_gap' else 50_000_000))
    assert owner.evidence.fault==('release_pose_joint_sequence_gap' if damage=='sequence_jump'
                                  else 'release_pose_joint_gap_or_reversal')
    assert n._cancel.is_set()


def test_successful_actual_open_clears_target_then_preserves_identity_through_finish(monkeypatch):
    n,owner=fixture(monkeypatch,opened=False);actual=actual_methods()
    original_target=n._target_book_model
    n._payload_monitor_enabled=True;n._retention_probe_active=False
    n._command_gripper=lambda target,**options:n.commands.append((target,options)) or True
    n._clear_target_contact_samples_unlocked=lambda **options:actual[
        '_clear_target_contact_samples_unlocked'](n,**options)
    def measured():
        fields=real_open_measurement(n,monkeypatch)
        assert n._held_book_corners is not None and n._target_book_model==original_target
        return finish.remember_open(n,owner,fields)
    assert actual['_open_gripper'](n,verify_measurement=measured)
    assert n._held_book_corners is None and n._target_book_model is None
    assert n._gripper_open_confirmed and not n._payload_monitor_enabled
    assert owner.binding.identity.target_model==original_target
    assert owner.binding.guard is n._place_contact_guard
    ready(n,owner)
    assert finish.publish_terminal(n,vars(IDENTITY))
    assert [event for event,_ in n.events]==[
        'placement_open_measured','placement_release_pose_measured','succeeded']
    assert n.events[-1][1]['hand_return'] is False
    assert n.commands==[(.069,dict(respect_cancel=True))]
    finish.deactivate(n)
    assert n._release_pose_owner is None and n._release_pose_fault_latched is None



def actual_checked_return_plan(mode):
    """Exercise the original empty-scene acceptance block and actual result type.

    Scene.leg is a controlled predicate here. This checks success-marker wiring,
    not physical geometry, IK, or an unexecuted complete planner route.
    """
    import copy
    from erc_phase1_solution import scene_checked_place as planner
    path = Path(planner.__file__)
    tree = ast.parse(path.read_text())
    selected = []
    for parent in ast.walk(tree):
        for _, value in ast.iter_fields(parent):
            if not isinstance(value, list):
                continue
            for index, statement in enumerate(value):
                if (isinstance(statement, ast.Assign)
                        and len(statement.targets) == 1
                        and isinstance(statement.targets[0], ast.Name)
                        and statement.targets[0].id == 'returns'
                        and isinstance(statement.value, ast.List)
                        and not statement.value.elts):
                    selected.append(copy.deepcopy(value[index:]))
    assert len(selected) == 1
    block = selected[0]
    assert len(block) == 4
    # Keep both original options and every original per-leg predicate/loop.
    assert isinstance(block[-1], ast.For)
    wrapper = ast.parse('def select():\n    accepted = None\n    def candidate():\n        nonlocal accepted\n    candidate()\n    return accepted').body[0]
    wrapper.body[1].body.extend(block)
    calls = []
    def leg(start, goal, aperture, carried):
        calls.append((start, goal, aperture, carried))
        return not (mode == 'direct_rejected_fallback' and len(calls) == 1)
    solutions = [tuple(GOAL), tuple(GOAL)]
    route = [tuple(GOAL)]
    scope = dict(solutions=solutions, home=tuple(GOAL), HOME=tuple(GOAL),
        table=None if mode == 'fallback_only' else object(), route=route,
        torso_ready=tuple(GOAL), unloaded=[], open_aperture=.069,
        scene=NS(leg=leg,samples=0), progress=lambda *a, **k:None,
        candidate_number=1, policy={})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),
                 str(path), 'exec'), scope)
    accepted = scope['select']()
    assert accepted is not None and calls and all(c[3] is False for c in calls)
    route, policy, return_legs, direct_home = accepted
    assert return_legs > 0
    plan = planner.SceneCheckedPlacePlan(solutions, 0, 0., route, [],
        {'direct_empty_fold_from_clearance': direct_home is not None}, direct_home)
    assert type(plan) is planner.SceneCheckedPlacePlan
    return plan


@pytest.mark.parametrize('mode', ['direct_checked', 'fallback_only', 'direct_rejected_fallback'])
def test_actual_checked_plan_result_and_place_call_admit_only_direct_marker(monkeypatch, mode):
    import copy
    plan = actual_checked_return_plan(mode)
    n, _ = fixture(monkeypatch, opened=False)
    finish.deactivate(n)
    assert n._release_pose_owner is None
    path = Path(__file__).resolve().parents[1]/'erc_phase1_solution/manipulation_node.py'
    cls = next(x for x in ast.parse(path.read_text()).body if isinstance(x, ast.ClassDef)
               and x.name == 'ManipulationNode')
    place = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '_place')
    assignments = {}
    for item in ast.walk(place):
        if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
            if item.targets[0].id in ('direct_empty_home', 'release_owner'):
                # Ignore the initial None assignment; retain the actual plan field extraction.
                if item.targets[0].id == 'direct_empty_home' and not isinstance(item.value, ast.Call):
                    continue
                assert item.targets[0].id not in assignments
                assignments[item.targets[0].id] = copy.deepcopy(item)
    assert set(assignments) == {'direct_empty_home', 'release_owner'}
    scope = dict(self=n, scene_plan=plan, correlation=vars(IDENTITY),
                 solutions=plan.solutions, release_pose_finish=finish)
    initialization = copy.deepcopy(place.body[0])
    assert isinstance(initialization, ast.Assign)
    assert [target.id for target in initialization.targets] == ['release_only_request', 'release_only_endpoint']
    assert isinstance(initialization.value, ast.Constant) and initialization.value.value is None
    code = compile(ast.fix_missing_locations(ast.Module(body=[initialization, assignments['direct_empty_home'],
        assignments['release_owner']], type_ignores=[])), str(path), 'exec')
    if mode == 'direct_checked':
        assert plan.empty_return_from_clearance == []
        exec(code, scope)
        assert scope['direct_empty_home'] is plan.empty_return_from_clearance
        assert scope['release_owner'] is n._release_pose_owner
        assert n._release_pose_owner.binding.identity == IDENTITY
    else:
        assert plan.empty_return_from_clearance is None
        assert plan.diagnostics['direct_empty_fold_from_clearance'] is False
        with pytest.raises(RuntimeError, match='complete_checked_normal_return_plan'):
            exec(code, scope)
        assert n._release_pose_owner is None
    assert not n.commands and not n.events


@pytest.mark.parametrize('marker', [(GOAL,), [GOAL], False])
def test_unsupported_direct_result_shapes_cannot_replace_checked_empty_marker(monkeypatch, marker):
    n, _ = fixture(monkeypatch, opened=False)
    finish.deactivate(n)
    with pytest.raises(RuntimeError, match='complete_checked_normal_return_plan'):
        finish.prepare(n, vars(IDENTITY), GOAL, marker)
    assert n._release_pose_owner is None and not n.commands
