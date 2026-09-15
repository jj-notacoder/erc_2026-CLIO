"""Prepared tests: real contract plus AST-extracted actual node methods, no ROS."""
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from erc_phase1_solution import marker_search_completion as m

ROOT = Path(__file__).resolve().parents[1]
TRIAL = '0123456789ab'


def request(epoch=1, before=1_000_000_000):
    return m.SearchRequest(TRIAL, epoch, 2, 'red', before)


def event(sequence, *, epoch=1, stamp=None, outcome='target_absent', count=0):
    stamp = stamp or 1_100_000_000+sequence*70_000_000
    return dict(request(epoch).fields(), event='marker_search_frame_completed', mode='markers', schema=1,
        frame_sequence=sequence, rgb_stamp_ns=stamp, depth_stamp_ns=stamp,
        completed_ros_ns=stamp+10_000_000, frame_valid=True, outcome=outcome, target_candidate_count=count,
        frame_age_limit_ns=m.MAX_AGE_NS, frame_valid_until_ns=stamp+m.MAX_AGE_NS)


def observe(gate, data):
    gate.observe(data, data['completed_ros_ns'])


def ready_gate():
    gate=m.NegativeFrameGate(request())
    for seq in (1,2,3): observe(gate,event(seq))
    return gate


def test_requires_three_completed_fresh_distinct_frames():
    gate=m.NegativeFrameGate(request())
    for seq in (1,2):
        observe(gate,event(seq)); assert not gate.ready(event(seq)['completed_ros_ns'])
    observe(gate,event(3)); assert gate.ready(event(3)['completed_ros_ns'])
    assert not gate.ready(gate.last_rgb_ns+m.MAX_AGE_NS+1)
    assert not gate.ready(gate.last_completed_ns-1)


@pytest.mark.parametrize('field,value', [
    ('schema',True),('schema',2),('trial_id','different'),('marker_search_request_id',2),
    ('marker_search_request_id',True),('shelf_column_number',3),('book_colour','blue'),
    ('marker_search_not_before_ns',True),('marker_search_not_before_ns',0),
    ('event','ready'),('mode','books'),('frame_sequence',True),('frame_sequence',1.5),
    ('rgb_stamp_ns',None),('rgb_stamp_ns',float('nan')),('depth_stamp_ns',0),
    ('completed_ros_ns',True),('frame_valid',1),('frame_valid',False),
    ('outcome',[]),('outcome','unknown'),('target_candidate_count',False),
    ('target_candidate_count',-1),
])
def test_invalid_metadata_never_completes_a_streak(field,value):
    gate=m.NegativeFrameGate(request())
    observe(gate,event(1));observe(gate,event(2))
    bad=event(3);bad[field]=value
    gate.observe(bad,1_500_000_000)
    assert gate.count==0 and not gate.ready(1_500_000_000)


@pytest.mark.parametrize('outcome,count', [
    ('target_present_pending',1),('depth_rejected',1),('cloud_published',1),
    ('processing_error',0),('frame_invalid',0),('target_absent',1),
])
def test_nonnegative_results_reset_streak(outcome,count):
    gate=ready_gate();observe(gate,event(4,outcome=outcome,count=count))
    assert gate.count==0


@pytest.mark.parametrize('mutation', ['sequence','rgb','depth','reversed','before_view','future','skew','stale'])
def test_replayed_or_unfresh_frames_do_not_count(mutation):
    gate=m.NegativeFrameGate(request());observe(gate,event(1));observe(gate,event(2))
    data=event(3); now=data['completed_ros_ns']
    if mutation=='sequence': data['frame_sequence']=2
    if mutation=='rgb': data['rgb_stamp_ns']=event(2)['rgb_stamp_ns']
    if mutation=='depth': data['depth_stamp_ns']=event(2)['depth_stamp_ns']
    if mutation=='reversed': data['completed_ros_ns']=event(1)['completed_ros_ns']
    if mutation=='before_view': data['rgb_stamp_ns']=request().not_before_ns
    if mutation=='future': now=data['rgb_stamp_ns']-1
    if mutation=='skew': data['depth_stamp_ns']=data['rgb_stamp_ns']-m.MAX_SKEW_NS-1
    if mutation=='stale': now=data['rgb_stamp_ns']+m.MAX_AGE_NS+1
    gate.observe(data,now);assert gate.count==0


def test_missing_completion_resets_but_allows_three_later_frames():
    gate=m.NegativeFrameGate(request());observe(gate,event(1));observe(gate,event(3))
    assert gate.count==1
    observe(gate,event(4));assert gate.count==2
    observe(gate,event(5));assert gate.ready(event(5)['completed_ros_ns'])


def methods(filename, names, namespace=None):
    tree=ast.parse((ROOT/'erc_phase1_solution'/filename).read_bytes())
    owner=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    selected=[n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected}==set(names)
    cls=ast.ClassDef(name='Actual',bases=[],keywords=[],body=selected,decorator_list=[])
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[])
    from erc_phase1_solution import empty_head_timing as actual_empty_head_timing
    scope={'__package__':'erc_phase1_solution', '_empty_head_timing':actual_empty_head_timing, **(namespace or {})}
    exec(compile(ast.fix_missing_locations(module),filename,'exec'),scope)
    return scope['Actual']


def stamp(ns): return NS(sec=ns//1_000_000_000,nanosec=ns%1_000_000_000)


def perception(detections=(), stable=False, point=None, enabled=True, error=None):
    calls=[]
    def detect(*args,**kwargs):
        calls.append('detector_completed')
        if error: raise error
        return list(detections)
    cls=methods('perception_node.py', ['_process_markers','_process'], dict(
        time=NS(monotonic=lambda:0), detect_number_markers=detect, PointCloud=lambda:NS(points=[],channels=[]),
        ChannelFloat32=lambda **kw:NS(values=[],**kw), Point32=lambda **kw:NS(**kw)))
    node=cls();node.mode='markers';node.target_column=2;node.target_colour='red'
    node.latest_rgb=object();node.templates=object();node.marker_confidence=.48
    node.latest_rgb_message=NS(header=NS(stamp=stamp(1_500_000_000)))
    node.latest_depth_message=NS(header=NS(stamp=stamp(1_500_000_000)))
    node.maximum_frame_age=1.;node.maximum_frame_skew=.2;node._ready=lambda:True
    node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=1_510_000_000))
    node._marker_search_request=request() if enabled else None
    node.marker_history=deque(maxlen=3);node._stable=lambda history:stable
    node._header=lambda:node.latest_depth_message.header;node._deproject=lambda *a,**k:point
    node.saved_modes={'markers'};node.marker_pub=NS(publish=lambda cloud:calls.append(('cloud',cloud)))
    node._publish_status=lambda kind,**kw:calls.append(dict(event=kind,mode=node.mode,**kw))
    return node,calls


def detection(digit):return NS(digit=digit,confidence=.9,center=(10*digit,10))


@pytest.mark.parametrize('digits,stable,point,outcome', [
    ([],False,None,'target_absent'),([1,3,4],False,None,'target_absent'),
    ([2],False,None,'target_present_pending'),([1,2,3],False,None,'target_present_pending'),
    ([1,2,3],True,None,'depth_rejected'),([1,2,3],True,(1.,2.,3.),'cloud_published'),
])
def test_actual_marker_outcomes_after_processing(digits,stable,point,outcome):
    node,calls=perception([detection(d) for d in digits],stable,point)
    node._process_markers()
    assert calls[0]=='detector_completed'
    records=[x for x in calls if isinstance(x,dict)]
    assert len(records)==1 and records[0]['outcome']==outcome
    assert records[0]['frame_valid'] is True
    if outcome=='cloud_published': assert calls[1][0]=='cloud'


def test_actual_processing_exception_is_not_absence_and_identity_preserved():
    error=RuntimeError('detector failed');node,calls=perception(error=error)
    with pytest.raises(RuntimeError) as caught: node._process_markers()
    assert caught.value is error
    assert calls[-1]['outcome']=='processing_error' and calls[-1]['frame_valid'] is False


def test_default_marker_path_has_no_completion_status():
    node,calls=perception(enabled=False);node._process_markers()
    assert calls==['detector_completed']


def test_actual_invalid_frame_path_resets_without_running_detector():
    node,calls=perception();node._ready=lambda:False;node.last_waiting_report_wall=0
    node._process()
    assert len(calls)==1 and calls[0]['outcome']=='frame_invalid'
    assert calls[0]['frame_valid'] is False


@pytest.mark.parametrize('kind', ['not_ready','stale','changed_epoch','changed_target','publish_error'])
def test_producer_invalidations_fail_closed(kind):
    node,calls=perception(); frame=m.capture_frame(node)
    if kind=='not_ready': node._ready=lambda:False;frame=m.capture_frame(node)
    if kind=='stale': node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=3_000_000_000))
    if kind=='changed_epoch': node._marker_search_request=request(2)
    if kind=='changed_target': node.target_column=3
    if kind=='publish_error': node._publish_status=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('publish'))
    m.complete_frame(node,frame,'target_absent',0)
    assert not calls or calls[-1]['frame_valid'] is False


def test_receive_repeated_request_keeps_sequence_but_old_or_invalid_clears():
    node,calls=perception();node._marker_search_frame_sequence=7
    m.receive_request(node,request().fields());assert node._marker_search_frame_sequence==7
    m.receive_request(node,request(2).fields());assert node._marker_search_frame_sequence==0
    m.receive_request(node,request().fields());assert node._marker_search_request is None
    m.receive_request(node,request(3).fields());node.mode='books'
    m.receive_request(node,request(3).fields());assert node._marker_search_request is None


def mission(state='SEARCH_COLUMN', enabled=True, elapsed=1.):
    cls=methods('mission_manager.py', ['_tick','_perception_mode','_marker_search_negative_ready','_on_markers','_on_perception_status'],
        dict(time=NS(monotonic=lambda:0),decode_event=json.loads,stamp_to_nanoseconds=lambda s:s.sec*10**9+s.nanosec))
    node=cls();calls=[];node.state=state;node.finished=False;node.wall_started=0.;node.trial_timeout=1000
    node.trial_id=TRIAL;node.target_column=2;node.target_colour='red';node.marker_search_negative_enabled=enabled
    node._marker_search_negative_gate=ready_gate();node._elapsed_state=lambda:elapsed;node.marker_search_timeout=5.
    node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=1_400_000_000))
    node.marker_cloud=None;node.search_turns=0;node.max_search_turns=2;node.robot_pose=(0.,0.,0.);node.search_turn=1.
    node._navigate=lambda *a,**kw:calls.append(('navigate',a,kw))
    node._abort=lambda reason:calls.append(('abort',reason))
    def state_change(value):calls.append(('state',value));node.state=value
    node._set_state=state_change;node._log=lambda *a,**kw:calls.append(('log',a,kw))
    node._nav_reached=lambda:True;node._manip_succeeded=lambda command:True
    node.mode_pub=object();node.row_confirmed=False;node.detected_row=None
    node._command=lambda pub,mode,**fields:calls.append(('mode',mode,fields));node.ready={'perception':False}
    return node,calls


def test_actual_tick_early_turn_uses_existing_navigation_and_turn_count():
    node,calls=mission();node._tick()
    assert node.search_turns==1 and node.state=='SEARCH_TURN'
    assert [c[1] for c in calls if c[0]=='navigate']==[(0.,0.,1.,'visual_search_turn')]
    assert calls[0][0]=='log' and calls[0][1]==('marker_search_early_turn',)


@pytest.mark.parametrize('case', ['positive','limit','missing_pose','disabled','unready','stale'])
def test_existing_positive_priority_limits_and_no_evidence_wait(case):
    node,calls=mission()
    if case=='positive':
        node.marker_cloud=NS(header=NS(stamp=stamp(1_300_000_000)))
        node._shelf_geometry=lambda cloud:('target','normal',3);node._republish_score=lambda:None
        node._goal_from_target=lambda *a:(1.,2.,3.);node.shelf_standoff=.8
    if case=='limit':node.search_turns=2
    if case=='missing_pose':node.robot_pose=None
    if case=='disabled':node.marker_search_negative_enabled=False
    if case=='unready':node._marker_search_negative_gate.reset()
    if case=='stale':node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=3_000_000_000))
    node._tick()
    if case=='positive':assert node.state=='NAVIGATE_SHELF' and not any(c[0]=='log' for c in calls)
    elif case in ('limit','missing_pose'):assert calls==[('abort','shelf_markers_not_found')]
    else:assert not calls and node.state=='SEARCH_COLUMN'


@pytest.mark.parametrize('enabled', [False,True])
def test_five_second_fallback_unchanged(enabled):
    node,calls=mission(enabled=enabled,elapsed=5.1);node._marker_search_negative_gate.reset();node._tick()
    assert node.search_turns==1 and node.state=='SEARCH_TURN'
    assert not any(c[0]=='log' for c in calls)


def test_each_actual_search_entry_renews_epoch_and_clears_prior_streak():
    node,calls=mission('HEAD_MARKERS');node._tick()
    first=node._marker_search_negative_gate.request
    assert node.state=='SEARCH_COLUMN' and node._marker_search_negative_gate.count==0
    node.state='SEARCH_TURN';node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=2_000_000_000));node._tick()
    second=node._marker_search_negative_gate.request
    assert second.request_id==first.request_id+1 and second.not_before_ns>first.not_before_ns
    assert len([c for c in calls if c[0]=='mode'])==2


def test_actual_status_callback_and_positive_cloud_reset():
    node,calls=mission();node._marker_search_negative_gate=m.NegativeFrameGate(request())
    for seq in (1,2,3):node._on_perception_status(NS(data=json.dumps(event(seq))))
    assert node._marker_search_negative_ready()
    node._on_markers(object());assert not node._marker_search_negative_ready()
    node.state='SEARCH_TURN';node._on_perception_status(NS(data=json.dumps(event(4))))
    assert node._marker_search_negative_gate.count==0


def test_defaults_and_complete_node_inverses():
    record=json.loads((ROOT/'test/fixtures/marker_search_inverse.json').read_bytes())
    for name,entry in record.items():
        from candidate_composition_support import restore_current_extensions
        data=restore_current_extensions((ROOT/name).read_bytes(),Path(name).name)
        for change in reversed(entry['changes']):
            assert data.count(change['new'].encode())==1
            data=data.replace(change['new'].encode(),change['old'].encode())
        assert hashlib.sha256(data).hexdigest()==entry['parent_sha256']
    tree=ast.parse((ROOT/'erc_phase1_solution/mission_manager.py').read_bytes())
    owner=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    declare=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='_declare_parameters')
    values=next(n.value for n in declare.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='values' for t in n.targets))
    index=next(i for i,k in enumerate(values.keys) if isinstance(k,ast.Constant) and k.value=='marker_search_negative_enabled')
    assert isinstance(values.values[index],ast.Constant) and values.values[index].value is False


def limited_event(sequence, stamp_ns, age_limit_ns=100_000_000):
    data=event(sequence,stamp=stamp_ns)
    data.update(frame_age_limit_ns=age_limit_ns,
                frame_valid_until_ns=min(data['rgb_stamp_ns'],data['depth_stamp_ns'])+age_limit_ns)
    return data


@pytest.mark.parametrize('configured,expected', [(.1,100_000_000),(1.,m.MAX_AGE_NS),(2.,m.MAX_AGE_NS)])
def test_actual_producer_carries_effective_configured_expiry(configured,expected):
    node,calls=perception();node.maximum_frame_age=configured
    node._process_markers()
    record=calls[-1]
    assert record['frame_valid'] is True
    assert record['frame_age_limit_ns']==expected
    assert record['frame_valid_until_ns']==1_500_000_000+expected
    gate=m.NegativeFrameGate(request())
    gate.observe(record,record['completed_ros_ns'])
    assert gate.count==1


def test_subnanosecond_configured_age_cannot_supply_a_zero_limit_certificate():
    node,calls=perception();node.maximum_frame_age=1e-10
    node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=1_500_000_000))
    node._process_markers()
    assert calls[-1]['frame_age_limit_ns']==0
    assert calls[-1]['frame_valid'] is False


@pytest.mark.parametrize('field,value', [
    ('frame_age_limit_ns',None),('frame_age_limit_ns',False),('frame_age_limit_ns',0),
    ('frame_age_limit_ns',-1),('frame_age_limit_ns',1.5),('frame_age_limit_ns',float('nan')),
    ('frame_age_limit_ns',m.MAX_AGE_NS+1),('frame_age_limit_ns',m.MAX_STAMP_NS),
    ('frame_valid_until_ns',None),('frame_valid_until_ns',True),('frame_valid_until_ns',0),
    ('frame_valid_until_ns',-1),('frame_valid_until_ns',1.5),('frame_valid_until_ns',float('inf')),
    ('frame_valid_until_ns',m.MAX_STAMP_NS+1),('frame_valid_until_ns',1_510_000_000),
])
def test_missing_malformed_or_inconsistent_expiry_resets_streak(field,value):
    gate=m.NegativeFrameGate(request());observe(gate,event(1));observe(gate,event(2))
    data=event(3)
    if value is None:data.pop(field)
    else:data[field]=value
    gate.observe(data,data['completed_ros_ns'])
    assert gate.count==0 and not gate.ready(data['completed_ros_ns'])


def test_smaller_limit_rejects_delayed_completion_before_global_one_second_limit():
    gate=m.NegativeFrameGate(request())
    observe(gate,limited_event(1,1_100_000_000));observe(gate,limited_event(2,1_120_000_000))
    data=limited_event(3,1_140_000_000)
    assert 1_350_000_000-data['rgb_stamp_ns']<m.MAX_AGE_NS
    gate.observe(data,1_350_000_000)
    assert gate.count==0 and not gate.ready(1_350_000_000)


def test_new_completion_discards_streak_when_an_earlier_frame_has_expired():
    gate=m.NegativeFrameGate(request())
    observe(gate,limited_event(1,1_100_000_000));observe(gate,limited_event(2,1_150_000_000))
    observe(gate,limited_event(3,1_210_000_000))
    assert gate.count==1 and not gate.ready(1_220_000_000)
    observe(gate,limited_event(4,1_240_000_000));observe(gate,limited_event(5,1_270_000_000))
    assert gate.count==3 and gate.ready(1_280_000_000)


def test_turn_rechecks_oldest_expiry_even_when_latest_frame_remains_fresh():
    gate=m.NegativeFrameGate(request())
    for seq,stamp_ns in enumerate((1_100_000_000,1_120_000_000,1_140_000_000),1):
        observe(gate,limited_event(seq,stamp_ns))
    assert gate.ready(1_200_000_000)
    assert not gate.ready(1_200_000_001)
    assert 1_200_000_001<1_140_000_000+100_000_000
    assert not gate.ready(1_149_999_999)


def test_each_retained_frame_keeps_its_own_limit_when_later_limit_is_longer():
    gate=m.NegativeFrameGate(request())
    observe(gate,limited_event(1,1_100_000_000,100_000_000))
    observe(gate,limited_event(2,1_120_000_000,m.MAX_AGE_NS))
    observe(gate,limited_event(3,1_140_000_000,m.MAX_AGE_NS))
    assert gate.ready(1_150_000_000)
    assert not gate.ready(1_200_000_001)


def test_older_depth_stamp_sets_expiry_and_future_metadata_stays_rejected():
    gate=m.NegativeFrameGate(request());data=limited_event(1,1_150_000_000)
    data['depth_stamp_ns']=1_100_000_000
    data['frame_valid_until_ns']=1_200_000_000
    gate.observe(data,1_160_000_000);assert gate.count==1
    data=limited_event(2,1_180_000_000)
    gate.observe(data,1_179_999_999)
    assert gate.count==0


def test_actual_mission_tick_does_not_turn_after_configured_expiry():
    node,calls=mission();node._marker_search_negative_gate=m.NegativeFrameGate(request())
    now=[0];node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=now[0]))
    for seq,stamp_ns in enumerate((1_100_000_000,1_120_000_000,1_140_000_000),1):
        data=limited_event(seq,stamp_ns);now[0]=data['completed_ros_ns']
        node._on_perception_status(NS(data=json.dumps(data)))
    assert node._marker_search_negative_ready()
    now[0]=1_200_000_001
    node._tick()
    assert calls==[] and node.state=='SEARCH_COLUMN' and node.search_turns==0
