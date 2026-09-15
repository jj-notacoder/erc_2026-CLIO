"""Actual bounded negative solver using fake kinematics, no stored solution."""
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from erc_phase1_solution import scene_cartesian_solver as solver
from test_scene_cartesian_solver import Chain, Scene, ROTATION


def run(chain, *, limit=512, validator=lambda path, transition: True, diagnostics=None,
        proposals=None):
    node = NS(chain=chain, _cancel=threading.Event(), place_joint_limit_margin=.1,
              carried_supported_jaw_vertical_component=.75)
    seed = np.array([.3, .1, 0., -1., .2, 0., 0., -1.])
    positions = [[.1, 0., .2], [.2, 0., .2], [.3, 0., .2]]
    if proposals is None:
        proposals = (positions, [[x, .025, z] for x, _, z in positions])
    return solver.solve_scene_cartesian(node, positions, ROTATION, .3, seed, Scene(), validator,
        measured_seed=seed, aperture=.017, open_aperture=.019,
        limits=solver.SearchLimits(max_ik_calls=limit), diagnostics_out=diagnostics,
        position_proposals=proposals)


def test_negative_registered_targets_remain_center_first_and_exact_selected_index():
    chain = Chain()
    chain.modifier = lambda q, matrix, fixed: None if matrix[1, 3] == 0. else q
    detail = {}
    path, *_ = run(chain, diagnostics=detail)
    assert chain.calls[0][0][1, 3] == 0.
    assert detail['strategy']['position_proposal_index'] == 1
    assert detail['strategy']['anchor_ik_budget'] == 192
    assert all(q[-1] < 0. for q in path)
    assert all(chain.forward(q)[1, 3] == .025 for q in path)
    assert detail['candidates'] == 1 and detail['full_candidate_validator_required']


def test_negative_anchor_rejects_positive_ik_before_full_admission():
    chain = Chain(); validated = []
    def positive(q, matrix, fixed): q[-1] = 1.; return q
    chain.modifier = positive
    with pytest.raises(solver.SceneCartesianSearchError) as caught:
        run(chain, limit=18, validator=lambda *a: validated.append(a) or True)
    assert validated == []
    assert caught.value.diagnostics['rejections']['nonnegative_wrist'] > 0
    assert len(chain.calls) <= 18


def test_anchor_and_ordinary_fallback_share_one_small_ik_budget_and_clear_index():
    chain = Chain(); chain.modifier = lambda *a: None
    with pytest.raises(solver.SceneCartesianSearchError) as caught:
        run(chain, limit=8)
    detail = caught.value.diagnostics
    assert detail['ik_calls'] == len(chain.calls) == 8
    assert detail['strategy']['anchor_ik_budget'] == 4
    assert detail['strategy']['position_proposal_index'] is None
    assert detail['candidates'] == 0


def test_failed_anchor_does_not_label_ordinary_fallback_with_registered_index():
    chain = Chain()
    # All three anchor roots fail (ordinary seed, mirrored seed, bounded
    # template); later loose/tight fallback remains its own original target.
    def reject_anchor(q, matrix, fixed):
        if chain.calls[-1][2]['position_tolerance'] == .002 and matrix[1, 3] == .025:
            return None
        return q
    chain.modifier = reject_anchor
    detail = {}
    path, *_ = run(chain, diagnostics=detail,
        proposals=([[.1, .025, .2], [.2, .025, .2], [.3, .025, .2]],))
    assert detail['strategy']['position_proposal_index'] is None
    assert all(chain.forward(q)[1, 3] == 0. for q in path)
