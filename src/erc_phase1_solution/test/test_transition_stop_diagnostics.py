"""Diagnose actual failed predicates without changing stationary admission."""
import pytest
from erc_phase1_solution import place_transition_stop as stop
from test_place_transition_stop import model
from test_preopen_stationary import NAMES


@pytest.mark.parametrize('joint,field,offset,reason',[
    (NAMES[0],'positions',.00101,'arm_not_at_stationary_checked_endpoint'),
    (NAMES[1],'positions',.00201,'arm_not_at_stationary_checked_endpoint'),
    (NAMES[2],'velocities',.00101,'arm_not_at_stationary_checked_endpoint'),
    (NAMES[-1],'positions',.00051,'master_not_at_stationary_closed_reference'),
    (NAMES[-1],'velocities',.000101,'master_not_at_stationary_closed_reference')])
def test_failed_predicate_and_raw_residual_are_preserved_at_original_deadline(model,joint,field,offset,reason):
    f=model;entered=f.n.now
    def mutate(sample):sample[field][joint]+=offset
    f.n.mutate=mutate
    with pytest.raises(stop.TransitionStopRejected,match='transition_stop_deadline'):
        f.token.qualify()
    event=f.n.events[-1][1];details=event['stationarity_diagnostic']
    assert event['elapsed_ros_ns']==500_000_000
    assert details['accepted_samples']==0 and details['reason_counts'][reason]>0
    assert details['samples_evaluated']>=12 and len(details['recent_samples'])==12
    key='maximum_abs_position_error' if field=='positions' else 'maximum_abs_velocity'
    assert details[key][joint]==pytest.approx(offset)
    last=details['recent_samples'][-1]
    assert last['reason']==reason and not last['accepted']
    assert last['producer_stamp_ns']>entered
    index=details['joint_names'].index(joint)
    assert last['position_errors' if field=='positions' else 'velocities'][index]==pytest.approx(offset)
    assert not f.token.published and not f.n._delivery_measurement_active


def test_success_diagnostics_do_not_add_a_wait_or_change_fresh_window(model):
    f=model;f.token.qualify()
    details=f.n.events[-1][1]['stationarity_diagnostic']
    assert f.n.now==1_125_000_000
    assert details['accepted_samples']==5
    assert details['reason_counts']=={'stationary_closed_at_checked_endpoint':5}
    assert f.token.qualified and not f.token.published
    f.token.close()
