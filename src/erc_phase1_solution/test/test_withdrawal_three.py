"""Optional cap3 software tests; no physical retention or acceleration claim."""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from erc_phase1_solution.withdrawal_timing import checked_withdrawal_speed_scale
from erc_phase1_solution.arm_velocity_admission import ArmVelocityAdmissionRejected
import test_withdrawal_speed as legacy
import test_optional_arm_timing as timing
import test_arm_velocity_admission as velocity

ROOT=Path(__file__).resolve().parents[1]
INVERSE=ROOT/'test/fixtures/withdrawal_three_inverse.json'
INVERSE_SHA='22de6da239a0185b3e0a54e71f42fe1bb29dd990ee2b43ba276efe0c80feec40'


def restore(data):
    assert hashlib.sha256(INVERSE.read_bytes()).hexdigest()==INVERSE_SHA
    record=json.loads(INVERSE.read_bytes())
    assert hashlib.sha256(data).hexdigest()==record['candidate_sha256']
    text=data.decode()
    for row in reversed(record['replacements']):
        assert text.count(row['new'])==row['occurrences']
        text=text.replace(row['new'],row['old'])
    assert text==record['parent_source']
    assert hashlib.sha256(text.encode()).hexdigest()==record['parent_sha256']
    return text.encode()


def test_full_helper_inverse_and_unchanged_scope_function():
    actual=(ROOT/'erc_phase1_solution/withdrawal_timing.py').read_bytes()
    parent=restore(actual)
    # The complete route/scope admission function is unchanged by the cap.
    marker=b'def withdrawal_leg_speed_scales('
    assert actual[actual.index(marker):]==parent[parent.index(marker):]


@pytest.mark.parametrize('damage',['missing','duplicate','unrelated'])
def test_strict_helper_inverse_rejects_unlisted_changes(damage):
    data=(ROOT/'erc_phase1_solution/withdrawal_timing.py').read_bytes()
    token=b'not 1.0 <= scale <= 3.0'
    if damage=='missing':data=data.replace(token,b'not 1.0 <= scale <= 2.0',1)
    elif damage=='duplicate':data+=token
    else:data+=b'\n# unrelated\n'
    with pytest.raises(AssertionError):restore(data)


@pytest.mark.parametrize('scale',[2.,2.25,2.5,math.nextafter(3.,1.),3.])
def test_newly_admitted_range_retains_exact_float(scale):
    assert checked_withdrawal_speed_scale(scale)==scale


@pytest.mark.parametrize('scale',[math.nextafter(3.,math.inf),3.000001,4.])
def test_above_three_remains_rejected(scale):
    with pytest.raises(ValueError,match=r'\[1.0, 3.0\]'):
        checked_withdrawal_speed_scale(scale)


@pytest.mark.parametrize('scale',[2.,2.5,3.])
def test_four_actual_locked_velocity_admissions_at_new_duration(scale):
    n,events,sent,records,_,clock=velocity.sender_node(retained=True)
    make=timing.load('_make_retained_arm_trajectory_goal')
    n._make_retained_arm_trajectory_goal=lambda route:make(n,route)
    sender=timing.load('_send_retained_arm_trajectory',time=clock)
    n._send_retained_arm_trajectory=lambda *a,**k:sender(n,*a,**k)
    n._fresh_retention_probe=lambda *a,**k:True
    n._retention_after_leg=lambda *a:True
    def pressure_send(request):
        with n.command:
            with n._lock:return request()
    assert legacy.execute()(n,legacy.legs(),'pick',withdrawal_speed_scale=scale,
        initial_pressure_gate=NS(send=pressure_send),fresh_retention_phases=('initial_shelf_lift',))==(True,5,False)
    assert len(sent)==5 and len(records)==4 and all(r['admitted'] for r in records)
    expected=int(5.8/scale*1_000_000_000)
    assert all(r['points'][0]['time_from_start_ns']==expected for r in records)
    assert all(r['producer_stamps_ns']==[velocity.NOW]*7 for r in records)


@pytest.mark.parametrize('fault',['fresh_slope','stale','cancel','hazard'])
def test_three_cannot_bypass_current_state_or_retention_veto(fault):
    n,events,sent,records,_,clock=velocity.sender_node(retained=True)
    make=timing.load('_make_retained_arm_trajectory_goal');count=0
    def make_goal(route):
        nonlocal count
        count+=1;goal=make(n,route)
        if count==2:
            if fault=='fresh_slope':n.joints[velocity.NAMES[0]]=-20.
            elif fault=='stale':n._joint_stamps_ns[velocity.NAMES[0]]=velocity.NOW-150_000_001
            elif fault=='cancel':n._cancel.set()
            else:n._payload_hazard_reason=lambda **kw:'contact_lost'
        return goal
    n._make_retained_arm_trajectory_goal=make_goal
    sender=timing.load('_send_retained_arm_trajectory',time=clock)
    n._send_retained_arm_trajectory=lambda *a,**k:sender(n,*a,**k)
    n._fresh_retention_probe=lambda *a,**k:True
    n._retention_after_leg=lambda *a:True
    def pressure_send(request):
        with n.command:
            with n._lock:return request()
    kwargs=dict(withdrawal_speed_scale=3.,initial_pressure_gate=NS(send=pressure_send),
        fresh_retention_phases=('initial_shelf_lift',))
    if fault in ('fresh_slope','stale'):
        with pytest.raises(RuntimeError,match='^pick_recovery_failed$') as caught:
            legacy.execute()(n,legacy.legs(),'pick',**kwargs)
        assert isinstance(caught.value.__cause__,ArmVelocityAdmissionRejected)
        assert len(records)==1 and not records[0]['admitted']
    else:
        assert legacy.execute()(n,legacy.legs(),'pick',**kwargs)==(False,1,fault=='hazard')
    assert len(sent)==1  # The unchanged first lift only; no unsafe withdrawal.
    assert not n._pending_retained_acceptances and not n._goal_handles


@pytest.mark.parametrize('change',[dict(arm_speed_scale=3.),dict(command='place'),dict(leg_offset=1)])
def test_three_preserves_unstacked_pick_only_scope(change):
    with pytest.raises(ValueError):legacy.scales(legacy.legs(),scale=3.,**change)
