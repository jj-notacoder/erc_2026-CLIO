from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from test_bottom_shelf_approach import context
from test_bottom_shelf_approach import draft as approach

from erc_phase1_solution import bottom_grasp_proposal as draft


@pytest.fixture
def proposal_context(context):
    c=context
    def interpolate(first,last,step):
        count=int(np.ceil(np.linalg.norm(last-first)/step))
        return list(np.linspace(first,last,count+1)[1:])
    c.node._interpolate_positions=Mock(side_effect=interpolate)
    def solve(positions, rotations, torso, **kwargs):
        values=[]
        for position in positions:
            q=c.grasp.copy();q[[1,2,4]]=position;values.append(q)
        return values,0,.123,[np.full(8,999.)]  # Old setup must be discarded.
    c.node._solve_cartesian_path=Mock(side_effect=solve)
    return c


def test_reproduces_original_solver_strategy_without_q_override(proposal_context):
    c=proposal_context
    proposal=draft.propose_bottom_grasp(c.node,c.front,measured_start=c.guard.start)
    args=c.node._solve_cartesian_path.call_args
    np.testing.assert_allclose(np.array(args.args[0])[:,0],
        [.45,.5025,.555,.6016666666667,.6483333333333,.695],atol=1e-12)
    assert args.args[2]==.35
    assert args.kwargs['endpoint_first'] is False
    assert args.kwargs['first_valid'] is False
    assert 'skip_setup_transition' not in args.kwargs
    assert 'candidate_validator' not in args.kwargs
    np.testing.assert_array_equal(args.kwargs['transition_start'],c.guard.start)
    assert not hasattr(proposal,'solutions') and not hasattr(proposal,'transition_waypoints')
    assert proposal.grasp_solution[1]==pytest.approx(.695)
    assert proposal.withdrawal_solutions[0][1]==pytest.approx(.6483333333333)
    assert not proposal.grasp_solution.flags.writeable
    c.node._move_arm_solution.assert_not_called()


def test_rebuild_is_mandatory_and_receives_exact_proposal(proposal_context,monkeypatch):
    c=proposal_context;seen=[]
    def rebuild(node,front,grasp,**kwargs):
        seen.append((grasp,kwargs))
        return NS(solutions=(grasp,),extraction_solutions=kwargs['withdrawal_solutions'])
    monkeypatch.setattr(approach,'plan_bottom_shelf_approach',rebuild)
    proposal,result=draft.propose_and_rebuild_bottom_approach(c.node,c.front,
        empty_guard=c.guard,bay=c.args['bay'])
    assert seen[0][0] is proposal.grasp_solution
    assert seen[0][1]['withdrawal_solutions'] is proposal.withdrawal_solutions
    assert seen[0][1]['empty_guard'] is c.guard and seen[0][1]['bay'] is c.args['bay']
    assert result.solutions[-1] is proposal.grasp_solution


def test_rebuild_failure_cannot_return_legacy_approach(proposal_context,monkeypatch):
    c=proposal_context
    monkeypatch.setattr(approach,'plan_bottom_shelf_approach',
        Mock(side_effect=RuntimeError('measured bay rejected')))
    with pytest.raises(RuntimeError,match='measured bay rejected'):
        draft.propose_and_rebuild_bottom_approach(c.node,c.front,empty_guard=c.guard,bay=c.args['bay'])
    c.node._move_arm_solution.assert_not_called()


def test_rebuild_cannot_substitute_withdrawal_branch(proposal_context,monkeypatch):
    c=proposal_context
    def rebuild(node,front,grasp,**kwargs):
        changed=kwargs['withdrawal_solutions'][0].copy();changed[3]+=.1
        return NS(solutions=(grasp,),extraction_solutions=(changed,))
    monkeypatch.setattr(approach,'plan_bottom_shelf_approach',rebuild)
    with pytest.raises(RuntimeError,match='changed_during_rebuild'):
        draft.propose_and_rebuild_bottom_approach(c.node,c.front,empty_guard=c.guard,bay=c.args['bay'])


def test_bad_numerical_solver_output_is_rejected(proposal_context):
    c=proposal_context;original=c.node._solve_cartesian_path.side_effect
    def wrong(*args,**kwargs):
        result=original(*args,**kwargs);result[0][-1][1]+=.01;return result
    c.node._solve_cartesian_path.side_effect=wrong
    with pytest.raises(RuntimeError,match='residual_or_step'):
        draft.propose_bottom_grasp(c.node,c.front,measured_start=c.guard.start)


def test_nonbottom_target_is_rejected_before_ik(proposal_context):
    c=proposal_context;c.front[2]=.929
    with pytest.raises(ValueError,match='outside_workspace'):
        draft.propose_bottom_grasp(c.node,c.front,measured_start=c.guard.start)
    c.node._solve_cartesian_path.assert_not_called()
