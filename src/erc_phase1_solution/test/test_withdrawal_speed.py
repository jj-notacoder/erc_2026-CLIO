"""Prepared timing/admission cases; not physical acceleration or retention proof."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from erc_phase1_solution.withdrawal_timing import (
    checked_withdrawal_speed_scale, withdrawal_leg_speed_scales,
)
from erc_phase1_solution.arm_velocity_admission import ArmVelocityAdmissionRejected
from candidate_composition_support import restore_withdrawal_speed_bytes, restore_supported_compact_floor_bytes
import test_optional_arm_timing as timing
import test_arm_velocity_admission as velocity

ROOT = Path(__file__).resolve().parents[1]


def legs(extra=False):
    values = [(np.full(8, .1), 1., 'initial_shelf_lift')]
    values.extend((np.full(8, .1 + i*.02), 5.8, 'extraction') for i in range(1,5))
    if extra: values.append((np.full(8, .3), .65, 'fixed_orientation_lowering'))
    return values


def scales(route, scale=1.25, **changes):
    args=dict(command='pick', speed_scale=scale, arm_speed_scale=1., leg_offset=0, initial_pressure_gate=object())
    args.update(changes)
    return withdrawal_leg_speed_scales(route, **args)


def execute():
    return timing.load('_execute_retained_arm_legs', withdrawal_leg_speed_scales=withdrawal_leg_speed_scales)


@pytest.mark.parametrize('value', [1., 1.25, 1.5, 2., 2.25, 2.5, 3.])
def test_allowed_range(value):
    assert checked_withdrawal_speed_scale(value) == value


@pytest.mark.parametrize('value', [0., .999999, math.nextafter(1., 0.), math.nextafter(3., math.inf),
                                  3.000001, math.inf, -math.inf, math.nan, True, False, None, 'bad'])
def test_invalid_range(value):
    with pytest.raises(ValueError): checked_withdrawal_speed_scale(value)


@pytest.mark.parametrize('change', ['short', 'sixth_extraction', 'first_duration', 'first_phase',
                                  'withdrawal_duration', 'withdrawal_phase', 'command', 'stacked',
                                  'offset', 'no_pressure_gate'])
def test_wrong_route_or_scope_fails_before_dispatch(change):
    route=legs(); kwargs={}
    if change=='short': route.pop()
    elif change=='sixth_extraction': route.append(route[-1])
    elif change=='first_duration': route[0]=(route[0][0], .9, route[0][2])
    elif change=='first_phase': route[0]=(route[0][0], 1., 'extraction')
    elif change=='withdrawal_duration': route[2]=(route[2][0], 5.7, 'extraction')
    elif change=='withdrawal_phase': route[2]=(route[2][0], 5.8, 'recovery')
    elif change=='command': kwargs['command']='place'
    elif change=='stacked': kwargs['arm_speed_scale']=3.
    elif change=='offset': kwargs['leg_offset']=1
    else: kwargs['initial_pressure_gate']=None
    with pytest.raises(ValueError): scales(route, **kwargs)


@pytest.mark.parametrize('scale', [1.25, 1.5, 2., 2.25, 2.5, 3.])
def test_only_four_durations_change_and_all_q_objects_probes_watchdogs_remain(scale):
    route=legs(extra=True); saved=copy.deepcopy(route); pressure=object()
    n,calls,probes=timing.retained_node()
    assert execute()(n,route,'pick',withdrawal_speed_scale=scale,initial_pressure_gate=pressure,
                     fresh_retention_phases=('initial_shelf_lift',))==(True,6,False)
    assert len(calls)==6
    for i,call in enumerate(calls):
        expected=5.8/scale if 1<=i<=4 else saved[i][1]
        assert timing.seconds(call[0].trajectory.points[0].time_from_start)==pytest.approx(expected,abs=1e-9)
        assert call[0].trajectory.points[0].positions==list(saved[i][0][1:])
        assert call[1]==saved[i][1]  # original watchdog duration, including5.8
        assert call[2][0][0] is route[i][0]
        assert call[2][0][2]==saved[i][2]
        assert call[4].get('velocity_admission',False)==(1<=i<=4)
        assert ('initial_pressure_gate' in call[4])==(i==0)
        assert np.array_equal(route[i][0],saved[i][0]) and route[i][1:]==saved[i][1:]
    assert calls[0][4]['initial_pressure_gate'] is pressure
    assert probes[0]==('fresh',('pick','initial_shelf_lift'),{'leg':0})
    assert [p[1] for p in probes[1:]]==[('pick',phase,i) for i,(_,_,phase) in enumerate(route) if i]


def test_default_executor_matches_complete_parent_method_even_for_unrelated_phase():
    actual=execute(); parent_text=restore_withdrawal_speed_bytes((ROOT/'erc_phase1_solution/manipulation_node.py').read_bytes()).decode()
    owner=next(x for x in ast.parse(parent_text).body if isinstance(x,ast.ClassDef) and x.name=='ManipulationNode')
    method=next(x for x in owner.body if isinstance(x,ast.FunctionDef) and x.name=='_execute_retained_arm_legs')
    namespace=dict(actual.__globals__)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'<exact parent retained executor>','exec'),namespace)
    route=legs(extra=True); results=[]
    for function in (actual,namespace['_execute_retained_arm_legs']):
        n,calls,probes=timing.retained_node()
        result=function(n,route,'pick',initial_pressure_gate=None,fresh_retention_phases=('initial_shelf_lift',))
        described=[(c[0].trajectory.points[0].positions,timing.seconds(c[0].trajectory.points[0].time_from_start),c[1],c[2][0][2],c[4]) for c in calls]
        results.append((result,described,probes))
    assert results[0]==results[1]


def test_complete_source_inverse_and_sender_and_recovery_are_exact_parent():
    current=(ROOT/'erc_phase1_solution/manipulation_node.py').read_bytes()
    parent=restore_withdrawal_speed_bytes(current)
    declared=json.loads((ROOT/'test/fixtures/withdrawal_speed_inverse.json').read_bytes())
    assert hashlib.sha256(parent).hexdigest()==declared['parent_sha256']
    def methods(text):
        owner=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='ManipulationNode')
        return {n.name:ast.dump(n,include_attributes=False) for n in owner.body if isinstance(n,ast.FunctionDef)}
    old,new=methods(parent),methods(restore_supported_compact_floor_bytes(current))
    changed={name for name in old if old[name]!=new[name]}
    assert changed=={'__init__','_declare_parameters','_execute_retained_arm_legs','_pick'}
    assert old['_send_retained_arm_trajectory']==new['_send_retained_arm_trajectory']
    assert b'1.0 if lift_enabled and index == 0 else 5.8' in current


@pytest.mark.parametrize('damage', ['missing','duplicate','unrelated'])
def test_new_inverse_refuses_unlisted_changes(damage):
    data=(ROOT/'erc_phase1_solution/manipulation_node.py').read_bytes()
    fragment=b"            'withdrawal_speed_scale': 1.0,"
    if damage=='missing': data=data.replace(fragment,b'',1)
    elif damage=='duplicate': data+=fragment
    else: data+=b'\n# unrelated\n'
    with pytest.raises(AssertionError): restore_withdrawal_speed_bytes(data)


@pytest.mark.parametrize('fault', ['fresh_slope','stale','cancel','hazard'])
def test_each_shortened_send_uses_actual_fresh_state_with_existing_lock_and_stop(fault):
    n,events,sent,records,_,clock=velocity.sender_node(retained=True)
    make=timing.load('_make_retained_arm_trajectory_goal'); count=0
    def make_goal(route):
        nonlocal count
        count+=1
        goal=make(n,route)
        if count==2:
            if fault=='fresh_slope': n.joints[velocity.NAMES[0]]=-20.
            elif fault=='stale': n._joint_stamps_ns[velocity.NAMES[0]]=velocity.NOW-150_000_001
            elif fault=='cancel': n._cancel.set()
            else: n._payload_hazard_reason=lambda **kw:'contact_lost'
        return goal
    n._make_retained_arm_trajectory_goal=make_goal
    sender=timing.load('_send_retained_arm_trajectory',time=clock)
    n._send_retained_arm_trajectory=lambda *args,**kwargs:sender(n,*args,**kwargs)
    probes=[]
    n._fresh_retention_probe=lambda *args,**kwargs:(probes.append((args,kwargs)) or True)
    n._retention_after_leg=lambda *args:True
    def pressure_send(request):
        with n.command:
            with n._lock: return request()
    pressure=NS(send=pressure_send)
    kwargs=dict(withdrawal_speed_scale=2.,initial_pressure_gate=pressure,fresh_retention_phases=('initial_shelf_lift',))
    if fault in ('fresh_slope','stale'):
        with pytest.raises(RuntimeError, match='^pick_recovery_failed$') as caught:
            execute()(n,legs(),'pick',**kwargs)
        assert isinstance(caught.value.__cause__, ArmVelocityAdmissionRejected)
        assert len(records)==1 and not records[0]['admitted']
        if fault=='fresh_slope': assert records[0]['start_positions'][0]==-20.
    else:
        assert execute()(n,legs(),'pick',**kwargs)==(False,1,fault=='hazard')
    assert len(sent)==1  # only unchanged initial lift; no shortened goal published
    assert not n._pending_retained_acceptances and not n._goal_handles
    assert len(probes)==1


def test_all_four_shortened_sends_record_fresh_locked_velocity_admission():
    n,events,sent,records,_,clock=velocity.sender_node(retained=True)
    make=timing.load('_make_retained_arm_trajectory_goal')
    n._make_retained_arm_trajectory_goal=lambda route:make(n,route)
    sender=timing.load('_send_retained_arm_trajectory',time=clock)
    n._send_retained_arm_trajectory=lambda *args,**kwargs:sender(n,*args,**kwargs)
    n._fresh_retention_probe=lambda *args,**kwargs:True
    n._retention_after_leg=lambda *args:True
    def pressure_send(request):
        with n.command:
            with n._lock:return request()
    assert execute()(n,legs(),'pick',withdrawal_speed_scale=2.,initial_pressure_gate=NS(send=pressure_send),fresh_retention_phases=('initial_shelf_lift',))==(True,5,False)
    assert len(sent)==5 and len(records)==4 and all(r['admitted'] for r in records)
    assert all(r['points'][0]['time_from_start_ns']==2_900_000_000 for r in records)
    assert all(r['producer_stamps_ns']==[velocity.NOW]*7 for r in records)


def test_new_parameter_default_and_only_pick_forwarding():
    tree=ast.parse((ROOT/'erc_phase1_solution/manipulation_node.py').read_text())
    owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ManipulationNode')
    defaults=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='_declare_parameters')
    entries=[(k.value,v.value) for d in ast.walk(defaults) if isinstance(d,ast.Dict)
             for k,v in zip(d.keys,d.values) if isinstance(k,ast.Constant) and isinstance(v,ast.Constant)]
    assert ('withdrawal_speed_scale',1.) in entries
    forwarding=[n.name for n in owner.body if isinstance(n,ast.FunctionDef) for c in ast.walk(n)
                if isinstance(c,ast.Call) and any(k.arg is None and "'withdrawal_speed_scale'" in ast.unparse(k.value) for k in c.keywords)]
    assert forwarding==['_pick']


def test_withdrawal_only_constructor_loads_same_urdf_limits(tmp_path):
    path=tmp_path/'limits.urdf';names=('torso_lift_joint',*velocity.NAMES)
    path.write_text('<robot>'+''.join(f'<joint name="{name}"><limit velocity="{i+1}"/></joint>' for i,name in enumerate(names))+'</robot>')
    init=timing.method_ast('__init__')
    blocks=[n for n in init.body if isinstance(n,ast.If) and ast.unparse(n.test)=="self.placement_transport_speed_scale > 1.0 or self.withdrawal_speed_scale > 1.0 or self.empty_pickup_setup_retiming_enabled or getattr(self, 'bin_clearance_timing_enabled', False) or getattr(self, 'withdrawal_half_timing_enabled', False) or getattr(self, 'compact_extension_half_timing_enabled', False) or getattr(self, 'release_only_clearance_timing_enabled', False)"]
    assert len(blocks)==1
    n=NS(placement_transport_speed_scale=1.,withdrawal_speed_scale=1.25,empty_pickup_setup_retiming_enabled=False,bin_clearance_timing_enabled=False,withdrawal_half_timing_enabled=False)
    scope=dict(self=n,urdf=path,__name__='erc_phase1_solution.fixture',__package__='erc_phase1_solution')
    exec(compile(ast.fix_missing_locations(ast.Module(body=blocks,type_ignores=[])),'<actual withdrawal-only limits>','exec'),scope)
    assert n._faster_arm_velocity_limits==tuple(float(i) for i in range(2,9))
    assert n._faster_arm_velocity_urdf==str(path)


def test_optional_prefix_failure_aborts_before_any_closed_motion():
    n,calls,probes=timing.retained_node()
    route=legs(); route[2]=(route[2][0],5.7,'extraction')
    with pytest.raises(RuntimeError,match='^pick_recovery_failed$') as caught:
        execute()(n,route,'pick',withdrawal_speed_scale=2.,initial_pressure_gate=object())
    assert isinstance(caught.value.__cause__,ValueError)
    assert not calls and not probes


def test_new_veto_uses_actual_mission_abort_policy_even_with_retries_and_default_preserves_exception():
    veto=ArmVelocityAdmissionRejected('velocity_limit_exceeded')
    n,calls,probes=timing.retained_node(); sends=[]
    def send(*args,**kwargs):
        sends.append(args)
        if len(sends)==2: raise veto
        return True,False
    n._send_retained_arm_trajectory=send
    with pytest.raises(RuntimeError,match='^pick_recovery_failed$') as caught:
        execute()(n,legs(),'pick',withdrawal_speed_scale=2.,initial_pressure_gate=object())
    assert caught.value.__cause__ is veto and len(sends)==2
    reason=str(caught.value)
    mission_source=(ROOT/'erc_phase1_solution/mission_manager.py').read_bytes()
    # Pin the actual PICK policy executed below; PLACE handoff is independent.
    owner=next(n for n in ast.parse(mission_source).body if isinstance(n,ast.ClassDef) and n.name=='MissionManager')
    tick=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='_tick')
    blocks=[n for n in tick.body if isinstance(n,ast.If) and ast.unparse(n.test)=="self.state == 'PICK'"]
    assert len(blocks)==1
    assert hashlib.sha256(ast.get_source_segment(mission_source.decode(),blocks[0]).encode()).hexdigest()=='a6ad82eb43960072e23e53a6275b92df9fdc1372ffe30d881c0f4fac0a4a6412'
    fn=ast.parse('def pick_tick(self):\n    pass\n').body[0]
    fn.body=copy.deepcopy(blocks)
    scope={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<actual mission PICK policy>','exec'),scope)
    for maximum,event_reason,expected in ((1,reason,[('abort',reason)]),(3,reason,[('abort',reason)]),
            (3,'ordinary_failure',[('perception','idle'),('manipulate','look_book_row_2'),('state','RETRY_HEAD_BOOKS')])):
        events=[]
        mission=NS(state='PICK',pick_attempts=1,max_pick_attempts=maximum,detected_row=2,
            manip_event={'reason':event_reason},_manip_succeeded=lambda command:False,_manip_failed=lambda:True,
            _abort=lambda value:events.append(('abort',value)),
            _perception_mode=lambda value:events.append(('perception',value)),
            _manipulate=lambda value:events.append(('manipulate',value)),
            _set_state=lambda value:events.append(('state',value)))
        scope['pick_tick'](mission)
        assert events==expected
    # Legacy/default execution retains the original exception object and retry policy.
    n,calls,probes=timing.retained_node()
    def reject(*args,**kwargs): raise veto
    n._send_retained_arm_trajectory=reject
    with pytest.raises(ArmVelocityAdmissionRejected) as original:
        execute()(n,legs(),'pick')
    assert original.value is veto
