"""Same bounded solver and all endpoint/candidate gates for a positive carry."""
import numpy as np
import pytest
import test_scene_cartesian_solver as base
m = base.m


def prepared():
    t=base.SolverTests();t.setUp();t.seed[-1]=.9
    t.chain.modifier=lambda q,*args:np.r_[q[:-1], .9]
    return t


def test_default_retains_negative_only_admission():
    t=prepared()
    with pytest.raises(m.SceneCartesianSearchError):t.run_solver()
    assert not t.validations and not t.scene.calls


def test_positive_start_admits_only_fully_refined_checked_path():
    t=prepared();info={}
    path,*_=t.run_solver(wrist_policy='measured_positive',diagnostics_out=info)
    assert all(q[-1]>.0 for q in path) and t.validations
    assert info['search']=='bounded_signed_support_positive_start_v1'
    assert info['position_tolerance']==.002 and info['orientation_tolerance']==.02
    assert info['joint_margin']==.1 and info['minimum_signed_support']==.75
    assert info['ik_calls']<=512
    assert sorted({a for _,a,_ in t.scene.calls})==pytest.approx([.017,.018,.019])


@pytest.mark.parametrize('fault',['support','endpoint','opening','candidate','residual','margin'])
def test_positive_policy_preserves_each_admission_gate(fault):
    t=prepared()
    if fault=='support':t.chain.support=.74
    if fault=='endpoint':t.scene.accept=lambda *args:False
    if fault=='opening':t.scene.accept=lambda q,a,loaded:a<=.017
    if fault=='candidate':t.validator=lambda *args:False
    if fault=='residual':t.chain.modifier=lambda q,*args:np.r_[q[0],q[1]+.003,q[2:-1],.9]
    if fault=='margin':t.chain.modifier=lambda q,*args:np.r_[q[:-1],2.91]
    with pytest.raises(m.SceneCartesianSearchError):t.run_solver(wrist_policy='measured_positive')
    assert not t.validations


@pytest.mark.parametrize('policy',[None,True,'all','positive'])
def test_unknown_wrist_policy_rejected_before_ik(policy):
    t=prepared()
    with pytest.raises(ValueError):t.run_solver(wrist_policy=policy)
    assert not t.chain.calls


def test_positive_policy_cannot_be_requested_from_negative_measured_carry():
    t=prepared();t.seed[-1]=-.9
    with pytest.raises(ValueError,match='positive measured wrist'):
        t.run_solver(wrist_policy='measured_positive')
    assert not t.chain.calls


def test_positive_policy_does_not_raise_the_original_ik_limit():
    t=prepared()
    with pytest.raises(m.SceneCartesianSearchError,match='ik_budget') as error:
        t.run_solver(wrist_policy='measured_positive',limits=m.SearchLimits(max_ik_calls=1))
    assert error.value.diagnostics['ik_calls']==1 and not t.validations


def test_anchor_proposals_use_first_fully_admitted_target_with_one_shared_budget():
    t=prepared();info={}
    original=np.asarray(t.positions)
    nearer=original.copy();nearer[:,0]-=.1
    t.scene.accept=lambda q,*args:q[1]<.25
    path,*_=t.run_solver(wrist_policy='measured_positive',position_proposals=(original,nearer),diagnostics_out=info)
    assert info['strategy']['position_proposal_index']==1
    assert info['candidates']==1 and info['ik_calls']==len(t.chain.calls)
    np.testing.assert_allclose([[q[1],q[2],q[4]]for q in path],nearer)
    assert t.validations


def test_negative_default_cannot_receive_alternate_targets():
    t=prepared()
    with pytest.raises(ValueError,match='positive measured carry'):
        t.run_solver(position_proposals=(t.positions,))
    assert not t.chain.calls


def test_proposal_search_does_not_reset_global_ik_budget():
    t=prepared();t.scene.accept=lambda *args:False
    with pytest.raises(m.SceneCartesianSearchError,match='ik_budget')as error:
        t.run_solver(wrist_policy='measured_positive',position_proposals=(t.positions,t.positions),
                     limits=m.SearchLimits(max_ik_calls=7))
    assert error.value.diagnostics['ik_calls']==7 and len(t.chain.calls)==7


def test_anchor_execution_order_retains_every_supplied_waypoint():
    t=prepared();t.positions=[[.1,0.,.1],[.3,0.,.3],[.3,0.,.2],[.3,0.,.1]]
    path,*_=t.run_solver(wrist_policy='measured_positive')
    assert len(path)==4
    np.testing.assert_allclose([[q[1],q[2],q[4]]for q in path],t.positions)
    np.testing.assert_array_equal(t.chain.calls[0][0][:3,3],t.positions[1])


def test_rejected_proposal_then_original_fallback_clears_selected_target_index():
    t=prepared();info={};near=np.asarray(t.positions).copy();near[:,0]-=.1
    t.validator=lambda path,unused:path[-1][1]>.25
    path,*_=t.run_solver(wrist_policy='measured_positive',position_proposals=(near,),diagnostics_out=info)
    assert info['strategy']['anchor_complete_paths']==1
    assert info['strategy']['position_proposal_index'] is None
    assert info['candidates']==2
    np.testing.assert_allclose([[q[1],q[2],q[4]]for q in path],t.positions)


def test_local_anchor_allowance_preserves_global_budget_for_original_fallback():
    t=prepared();info={}
    many=np.array([[x,0.,.2]for x in np.linspace(.1,.2,64)])
    path,*_=t.run_solver(wrist_policy='measured_positive',position_proposals=(many,),
        limits=m.SearchLimits(max_ik_calls=120),diagnostics_out=info)
    assert info['strategy']['anchor_ik_budget']==60
    assert info['strategy']['anchor_complete_paths']==0
    assert info['strategy']['position_proposal_index'] is None
    assert info['ik_calls']==66 and len(t.chain.calls)==66
    np.testing.assert_array_equal(t.chain.calls[60][0][:3,3],t.positions[0])
    np.testing.assert_allclose([[q[1],q[2],q[4]]for q in path],t.positions)


def test_cancellation_in_anchor_candidate_cannot_return_an_admitted_path():
    t=prepared()
    def candidate(path,unused):t.node._cancel.set();return True
    t.validator=candidate
    with pytest.raises(m.SceneCartesianSearchError,match='cancelled'):
        t.run_solver(wrist_policy='measured_positive',position_proposals=(t.positions,))
    assert len(t.chain.calls)==3
