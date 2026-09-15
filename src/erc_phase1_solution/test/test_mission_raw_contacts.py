"""Stock serialization versus actual mission callbacks; no simulator required.

Run only in the coordinated sourced ROS validation environment. The installed
ROS serializer is the wire oracle; the complete existing decoder is unchanged.
No node initialization, subscriptions, executor, geometry or actuation occurs.
"""
import ast
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import pytest
from rclpy.serialization import deserialize_message, serialize_message
from ros_gz_interfaces.msg import Contact, Contacts

from erc_phase1_solution import mission_manager as mission
from erc_phase1_solution import mission_raw_contacts as adapter
from erc_phase1_solution import raw_contacts as raw
from erc_phase1_solution.release_evidence import AttemptIdentity, ReleaseEvidence
from erc_phase1_solution.release_pose_evidence import ReleasePoseEvidence, contract_fields
from erc_phase1_solution.runtime_utils import ContactEpisodeTracker
from raw_contacts_oracle_support import full_message, IDLEncoder, canonical

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'erc_phase1_solution/mission_manager.py'
OWNER = next(n for n in ast.parse(SOURCE.read_text()).body
             if isinstance(n, ast.ClassDef) and n.name == 'MissionManager')
METHODS = {n.name:n for n in OWNER.body if isinstance(n, ast.FunctionDef)}
MODEL = 'book_col_3_row_2_red'
BOOK = MODEL+'::book_link::collision'
BIN = 'erc_collection_bin::bin_link::collision'
ROBOT = 'tiago_pro::gripper_left_fingertip_left_link::collision'
TABLE = 'erc_table::table_link::collision'
OTHER = 'other::link::collision'
FLOOR = 'ground_plane::link::collision'
IDENTITY = AttemptIdentity('trial', 'attempt', MODEL)


def message(stamp, pairs):
    value = Contacts()
    value.header.stamp.sec, value.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
    value.header.frame_id = 'world'
    for first,second in pairs:
        contact = Contact()
        contact.collision1.name,contact.collision2.name = first,second
        value.contacts.append(contact)
    return value


def node(monkeypatch, *, evidence='release_pose', state='PLACE'):
    n = NS(state=state, finished=False, now=1_000_000_000,
        target_book_model=MODEL, target_colour='red', _mission_raw_contacts_failure=None,
        delivery_evidence_enabled=evidence is not None, dry_run=False,
        trial_id='trial', first_target_bin_contact=None, bin_contact_confirmed=False,
        collision_episodes=0, active_contacts=set(), events=[], commands=[], summaries=[],
        active_nav_pending=False, _place_input_request=None,
        nav_command_pub='navigation', manip_command_pub='manipulation',
        contact_tracker=ContactEpisodeTracker(separation_gap=.25, cooldown=1.))
    n.delivery_release_evidence = (None if evidence is None else
        (ReleasePoseEvidence if evidence=='release_pose' else ReleaseEvidence)(IDENTITY))
    n._delivery_release_snapshot = None
    n.get_clock = lambda:NS(now=lambda:NS(nanoseconds=n.now))
    n._expected_target_book_model = lambda:MODEL
    n._contact_pairs = mission.MissionManager._contact_pairs
    n._log = lambda event,**fields:n.events.append((event,fields))
    n._command = lambda publisher,event,**fields:n.commands.append((publisher,event,fields))
    n._write_summary = lambda success,reason,**kw:n.summaries.append((success,reason))
    n._set_state = lambda state,**fields:setattr(n,'state',state)
    n._perception_mode = lambda mode:n.commands.append(('perception',mode,{}))
    n._persist_navigation_run = lambda event:n.events.append(('navigation_terminal',event))
    for name in ('_on_contacts','_on_bin_contacts','_on_manipulation_status',
                 '_record_first_target_bin_contact','_abort','_complete','_tick'):
        setattr(n,name,MethodType(getattr(mission.MissionManager,name),n))
    monkeypatch.setattr(mission.time,'monotonic',lambda:10.)
    monkeypatch.setattr(mission,'datetime',NS(now=lambda tz:NS(
        isoformat=lambda **kw:'2026-09-14T00:00:00.000000+00:00')))
    return n


def freeze(value):
    if isinstance(value,dict):
        return tuple(sorted((key,freeze(item)) for key,item in value.items()))
    if isinstance(value,(set,frozenset)):
        return tuple(sorted(freeze(item) for item in value))
    if isinstance(value,(tuple,list)):
        return tuple(freeze(item) for item in value)
    if hasattr(value,'__dict__'):
        return freeze(vars(value))
    return value


def state(n):
    return freeze(dict(state=n.state,finished=n.finished,target=n.target_book_model,
        score=n.collision_episodes,active=n.active_contacts,tracker=n.contact_tracker,
        contact=n.first_target_bin_contact,bin_confirmed=n.bin_contact_confirmed,
        events=n.events,commands=n.commands,summaries=n.summaries,
        evidence=n.delivery_release_evidence,snapshot=n._delivery_release_snapshot))


class Pair:
    def __init__(self, monkeypatch, **kwargs):
        self.nodes=[node(monkeypatch,**kwargs),node(monkeypatch,**kwargs)]
        self.callbacks=[]
        for i,n in enumerate(self.nodes):
            callbacks={}
            for topic,method in (('/contacts',n._on_contacts),('/bin_contacts',n._on_bin_contacts)):
                callbacks[topic] = ((lambda data,method=method:method(deserialize_message(data,Contacts)))
                    if i==0 else adapter.mission_raw_contacts_callback(n,method,topic))
            self.callbacks.append(callbacks)

    def deliver(self,value,topic='/contacts',now=None,data=None):
        data=serialize_message(value) if data is None else data
        assert canonical(deserialize_message(data,Contacts))==canonical(raw.decode_contacts_cdr(data))
        for n,callbacks in zip(self.nodes,self.callbacks):
            if now is not None:n.now=now
            callbacks[topic](data)
        assert state(self.nodes[0])==state(self.nodes[1])

    def status(self,payload,now):
        for n in self.nodes:
            n.now=now
            n._on_manipulation_status(NS(data=json.dumps(payload)))
        assert state(self.nodes[0])==state(self.nodes[1])

    def open(self,epoch=1_000_000_000,now=None):
        self.status(dict(vars(IDENTITY),event='placement_open_measured',command='place',
            verified=True,producer_stamp_ns=epoch),epoch if now is None else now)


@pytest.mark.parametrize('value',[0,1,None,'true',[],{},.0])
def test_non_boolean_flag_cannot_enable_raw_subscription(value):
    with pytest.raises(ValueError):adapter.checked_enabled(value)


@pytest.mark.parametrize('enabled',[False,True])
def test_actual_constructor_branch_preserves_types_topics_order_qos_and_callbacks(monkeypatch,enabled):
    assert adapter.checked_enabled(enabled) is enabled
    defaults=next(n for n in METHODS['_declare_parameters'].body if isinstance(n,ast.Assign))
    entries={k.value:v for k,v in zip(defaults.value.keys,defaults.value.values)}
    assert entries['mission_raw_contacts_enabled'].value is False
    assignment=next(n for n in METHODS['__init__'].body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Attribute) and t.attr=='mission_raw_contacts_enabled' for t in n.targets))
    n=node(monkeypatch,evidence=None)
    n.get_parameter=lambda key:NS(value=enabled) if key=='mission_raw_contacts_enabled' else pytest.fail(key)
    scope=dict(self=n,_mission_raw_contacts=adapter,Contacts=Contacts,SENSOR_QOS=object())
    exec(compile(ast.Module(body=[assignment],type_ignores=[]),str(SOURCE),'exec'),scope)
    branch=next(n for n in METHODS['__init__'].body if isinstance(n,ast.If)
        and isinstance(n.test,ast.Attribute) and n.test.attr=='mission_raw_contacts_enabled')
    calls=[]
    n.create_subscription=lambda *args,**kw:calls.append((args,kw))
    exec(compile(ast.Module(body=[branch],type_ignores=[]),str(SOURCE),'exec'),scope)
    assert [args[1] for args,kw in calls]==['/contacts','/bin_contacts']
    for args,kw in calls:
        assert args[0] is Contacts and args[3] is scope['SENSOR_QOS']
        assert kw==({'raw':True} if enabled else {})
    value=message(n.now,[(ROBOT,TABLE)])
    calls[0][0][2](serialize_message(value) if enabled else value)
    assert n.collision_episodes==1
    assert n.events[-1][0]=='unexpected_contact'
    assert not n.commands and n._mission_raw_contacts_failure is None


@pytest.mark.parametrize('seed',[0,1,2])
@pytest.mark.parametrize('endian',['<','>'])
def test_complete_stock_schema_names_stamps_and_unused_fields_preserved(monkeypatch,seed,endian):
    pair=Pair(monkeypatch,evidence=None)
    value=full_message(seed,names=[BOOK,ROBOT,TABLE,OTHER,BIN,'µ棚'])
    encoded=IDLEncoder(endian).write(value)
    pair.deliver(value,data=encoded,now=10_000_000_000+seed)
    pair.deliver(value,topic='/bin_contacts',data=encoded)


def test_original_scoring_episodes_duplicates_direction_and_target_change(monkeypatch):
    pair=Pair(monkeypatch,evidence=None,state='PICK')
    pair.deliver(message(1_000_000_000,[(BOOK,ROBOT),(ROBOT,TABLE),(TABLE,ROBOT)]))
    assert pair.nodes[0].collision_episodes==1
    pair.deliver(message(1_050_000_000,[(ROBOT,TABLE)]),now=1_050_000_000)
    pair.deliver(message(1_400_000_000,[]),now=1_400_000_000)
    pair.deliver(message(2_500_000_000,[(TABLE,ROBOT)]),now=2_500_000_000)
    assert pair.nodes[0].collision_episodes==2
    for n in pair.nodes:n.target_book_model='book_col_4_row_2_red'
    pair.deliver(message(2_600_000_000,[(BOOK,BIN)]),topic='/bin_contacts',now=2_600_000_000)
    assert pair.nodes[0].first_target_bin_contact is None


@pytest.mark.parametrize('topic',['/contacts','/bin_contacts'])
def test_valid_wire_with_empty_contact_set_is_delivered_not_invented(monkeypatch,topic):
    pair=Pair(monkeypatch)
    pair.open()
    pair.deliver(message(1_100_000_000,[]),topic,now=1_100_000_000)
    assert not pair.nodes[0].delivery_release_evidence.contact_stamps
    assert pair.nodes[0].delivery_release_evidence.fault is None


@pytest.mark.parametrize('relative',[-1,0,1])
@pytest.mark.parametrize('topic',['/contacts','/bin_contacts'])
def test_delayed_open_handoff_preserves_exact_epoch_veto_boundary(monkeypatch,relative,topic):
    pair=Pair(monkeypatch)
    epoch=1_050_000_000
    pair.deliver(message(epoch+relative,[(BOOK,ROBOT)]),topic,now=1_100_000_000)
    pair.deliver(message(1_100_000_000,[(OTHER,FLOOR)]),topic,now=1_100_000_000)
    pair.open(epoch,now=1_100_000_000)
    assert (pair.nodes[0].delivery_release_evidence.fault is None)==(relative<=0)


@pytest.mark.parametrize('kind',['robot','robot_bin','robot_table','stale','future','reversed','zero','empty_name'])
def test_received_fault_semantics_match_stock_before_any_terminal(monkeypatch,kind):
    pair=Pair(monkeypatch);pair.open()
    pair.deliver(message(1_100_000_000,[(BOOK,BIN)]),'/bin_contacts',now=1_100_000_000)
    pairs={'robot':[(BOOK,ROBOT)],'robot_bin':[(ROBOT,BIN)],'robot_table':[(ROBOT,TABLE)],
           'empty_name':[(BOOK,'')]}.get(kind,[(BOOK,BIN)])
    stamp={'stale':1_100_000_001,'future':1_600_000_001,'reversed':1_050_000_000,'zero':0}.get(kind,1_400_000_000)
    now=1_400_000_002 if kind=='stale' else 1_500_000_000 if kind=='future' else 1_200_000_000 if kind=='reversed' else 1_400_000_000
    pair.deliver(message(stamp,pairs),'/bin_contacts',now=now)
    pair.deliver(message(now+1,[]),'/bin_contacts',now=now+1)
    evidence=pair.nodes[0].delivery_release_evidence
    assert evidence.fault is not None
    assert not evidence.evaluate(now+1,10.)['operation_completed']


def test_full_actual_release_status_and_positive_bin_window_complete_equally(monkeypatch):
    pair=Pair(monkeypatch);pair.open()
    for stamp in range(1_100_000_000,1_600_000_001,100_000_000):
        pair.deliver(message(stamp,[(BOOK,BIN)]),'/bin_contacts',now=stamp)
    fields=dict(vars(IDENTITY),command='place',completion_mode='release_pose',hand_return=False,**contract_fields())
    pair.status(dict(fields,event='placement_release_pose_measured',verified=True,
        release_measurement_stamp_ns=1_600_000_000,producer_stamp_ns=1_600_000_000,
        odom_producer_stamp_ns=1_600_000_000,stationary_start_ns=1_450_000_000,
        bin_contact_start_ns=1_100_000_000,bin_contact_last_ns=1_600_000_000,
        bin_contact_samples=6,bin_contact_max_gap_ns=100_000_000),1_600_000_000)
    pair.status(dict(fields,event='succeeded',release_terminal_stamp_ns=1_600_000_000),1_600_000_000)
    for n in pair.nodes:
        assert n.delivery_release_evidence.evaluate(n.now,10.)['operation_completed']
        n._complete()
        assert n.finished and n.state=='DONE' and n.summaries==[(True,'')]
    assert state(pair.nodes[0])==state(pair.nodes[1])


def test_ordinary_hand_return_policy_remains_required(monkeypatch):
    pair=Pair(monkeypatch,evidence='ordinary');pair.open()
    for stamp in range(1_100_000_000,1_600_000_001,100_000_000):
        pair.deliver(message(stamp,[(BOOK,BIN)]),'/bin_contacts',now=stamp)
    for n in pair.nodes:
        n._complete()
        assert not n.finished and not n.summaries
        assert n.delivery_release_evidence.return_measurement is None


def test_bin_contacts_without_confirmed_target_do_not_become_evidence(monkeypatch):
    pair=Pair(monkeypatch)
    for n in pair.nodes:n.target_book_model=None
    pair.deliver(message(1_100_000_000,[(BOOK,BIN)]),'/bin_contacts',now=1_100_000_000)
    assert not pair.nodes[0].delivery_release_evidence.contact_stamps
    assert pair.nodes[0].first_target_bin_contact is None
    for n in pair.nodes:n.target_book_model=MODEL
    pair.open(1_100_000_000)
    pair.deliver(message(1_200_000_000,[(BOOK,BIN)]),'/bin_contacts',now=1_200_000_000)
    assert pair.nodes[0].delivery_release_evidence.contact_stamps=={1_200_000_000}


def test_future_open_status_waits_for_original_clock_catchup(monkeypatch):
    pair=Pair(monkeypatch)
    pair.open(1_050_000_000,now=1_000_000_000)
    assert pair.nodes[0].delivery_release_evidence.open_epoch is None
    pair.deliver(message(1_100_000_000,[(BOOK,BIN)]),'/bin_contacts',now=1_100_000_000)
    for n in pair.nodes:
        assert n.delivery_release_evidence.open_epoch==1_050_000_000
        assert n.delivery_release_evidence.contact_stamps=={1_100_000_000}


@pytest.mark.parametrize('kind',['short','truncated','trailing','representation','options','wrong_type'])
@pytest.mark.parametrize('topic',['/contacts','/bin_contacts'])
def test_malformed_wire_permanently_aborts_without_contact_or_success(monkeypatch,kind,topic):
    n=node(monkeypatch)
    valid=serialize_message(message(n.now,[(BOOK,BIN)]))
    bad={'short':b'\0','truncated':valid[:-1],'trailing':valid+b'\0',
         'representation':b'\0\3'+valid[2:],'options':valid[:2]+b'\0\1'+valid[4:],
         'wrong_type':'not bytes'}[kind]
    callbacks={source:adapter.mission_raw_contacts_callback(n,
        n._on_contacts if source=='/contacts' else n._on_bin_contacts,source)
        for source in ('/contacts','/bin_contacts')}
    callbacks[topic](bad)
    assert n.finished and n.state=='ABORTED'
    assert n.delivery_release_evidence.fault=='raw_contacts_decode_failed'
    assert n.summaries==[(False,'raw_contacts_decode_failed')]
    assert n.commands==[('navigation','cancel',{}),('manipulation','cancel',{})]
    assert n.first_target_bin_contact is None and not n.bin_contact_confirmed
    assert n.collision_episodes==0 and not n.delivery_release_evidence.contact_stamps
    failure=n._mission_raw_contacts_failure
    assert failure.source_topic==topic and len(failure.raw_prefix_hex)<=512
    with pytest.raises(FrozenInstanceError):failure.reason='changed'
    for callback in callbacks.values():callback(valid);callback(bad)
    n._tick();n._complete()
    assert n._mission_raw_contacts_failure is failure
    assert len(n.commands)==2 and len(n.summaries)==1
    assert [event for event,fields in n.events]==['sensor_fault']


@pytest.mark.parametrize('phase',['WAIT_READY','PICK','RETURN_START','WAIT_BIN_CONTACT'])
def test_wire_fault_without_release_owner_still_stops_mission(monkeypatch,phase):
    n=node(monkeypatch,evidence=None,state=phase)
    n.active_nav_pending=True;n.active_nav_goal_number=7
    adapter.mission_raw_contacts_callback(n,n._on_contacts,'/contacts')(b'bad')
    assert n.finished and n.state=='ABORTED' and not n.active_nav_pending
    assert n.active_nav_terminal==dict(event='mission_aborted',goal_id=7,reason='raw_contacts_decode_failed')
    assert n.summaries==[(False,'raw_contacts_decode_failed')]
    assert n.commands==[('navigation','cancel',{}),('manipulation','cancel',{})]


@pytest.mark.parametrize('failure_at',['evidence','navigation','manipulation','diagnostic','summary'])
def test_failure_publication_cannot_resume_or_hide_other_cancel_attempt(monkeypatch,failure_at):
    n=node(monkeypatch)
    original=n._command
    def command(publisher,event,**fields):
        original(publisher,event,**fields)
        if publisher==failure_at:raise RuntimeError('publish failure')
    n._command=command
    if failure_at=='diagnostic':n._log=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('log failure'))
    if failure_at=='summary':n._write_summary=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('summary failure'))
    if failure_at=='evidence':
        n.delivery_release_evidence.evaluate=lambda *a:(_ for _ in ()).throw(RuntimeError('evidence failure'))
    callback=adapter.mission_raw_contacts_callback(n,n._on_contacts,'/contacts')
    if failure_at=='diagnostic':callback(b'bad')
    else:
        with pytest.raises(RuntimeError):callback(b'bad')
    assert n.finished and n.state=='ABORTED'
    assert n._mission_raw_contacts_failure.reason=='raw_contacts_decode_failed'
    assert {'navigation','manipulation'} <= {publisher for publisher,event,fields in n.commands}
    assert n.delivery_release_evidence.fault=='raw_contacts_decode_failed'
    before=list(n.commands)
    callback(serialize_message(message(n.now,[(BOOK,BIN)])));n._complete();n._tick()
    assert n.commands==before and n.first_target_bin_contact is None


def test_finished_trial_is_not_rewritten_by_later_wire_failure(monkeypatch):
    n=node(monkeypatch);n.finished=True;n.state='DONE';n.summaries=[(True,'')]
    adapter.mission_raw_contacts_callback(n,n._on_contacts,'/contacts')(b'bad')
    assert n.state=='DONE' and n.summaries==[(True,'')] and not n.commands
    assert n.delivery_release_evidence.fault is None
    assert n.events[0][0]=='sensor_fault'


def test_original_callback_exception_is_not_reported_as_decode_failure(monkeypatch):
    n=node(monkeypatch);original=RuntimeError('callback fault')
    def callback(message):raise original
    wrapped=adapter.mission_raw_contacts_callback(n,callback,'/contacts')
    with pytest.raises(RuntimeError) as result:wrapped(serialize_message(message(n.now,[])))
    assert result.value is original
    assert n._mission_raw_contacts_failure is None and not n.commands and not n.events


def test_only_mission_contact_topics_are_accepted(monkeypatch):
    with pytest.raises(ValueError):
        adapter.mission_raw_contacts_callback(node(monkeypatch),lambda msg:None,'/table_contacts')


def test_unused_vector_access_is_not_required_by_actual_mission_callbacks(monkeypatch):
    pair=Pair(monkeypatch,evidence=None)
    def forbidden(*args):raise AssertionError('unused numeric field materialized')
    monkeypatch.setattr(raw._NumericSequence,'__getitem__',forbidden)
    monkeypatch.setattr(raw._Vector3,'x',property(forbidden))
    value=full_message(1,names=[BOOK,ROBOT,TABLE,BIN])
    # Deliberately bypass Pair.deliver's complete canonical comparison, which
    # independently tests all fields elsewhere and therefore accesses vectors.
    data=serialize_message(value)
    for callbacks in pair.callbacks:
        callbacks['/contacts'](data);callbacks['/bin_contacts'](data)
    assert state(pair.nodes[0])==state(pair.nodes[1])
