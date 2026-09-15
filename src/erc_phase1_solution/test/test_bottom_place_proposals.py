"""Pure proposal math and actual Request identity; no official mesh or ROS."""
from dataclasses import replace
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from erc_phase1_solution import bottom_place_proposals as mod
from erc_phase1_solution import release_only_place_planning as policy
from erc_phase1_solution.place_contact_guard import PlaceContactGuard
from erc_phase1_solution.release_evidence import AttemptIdentity
from test_release_only_place_planning import request_node, PAYLOAD


def retained(monkeypatch, model='book_col_1_row_5_green'):
    node = request_node(monkeypatch)
    identity = dict(PAYLOAD, target_model=model)
    node._target_book_model = model
    node._place_contact_guard = PlaceContactGuard(10, identity)
    node._transport_lock_engaged = True
    node._gravity_supported_payload = True
    node._payload_monitor_enabled = True
    return node, policy.capture(node, identity)


@pytest.mark.parametrize('column', range(1, 6))
@pytest.mark.parametrize('color', ('red', 'green', 'yellow', 'blue'))
def test_every_canonical_bottom_identity_is_eligible(monkeypatch, column, color):
    node, request = retained(monkeypatch, f'book_col_{column}_row_5_{color}')
    assert mod.verified_bottom_request(node, request)
    assert node.context_checks > 1


@pytest.mark.parametrize('row', (2, 3, 4))
@pytest.mark.parametrize('column', range(1, 6))
@pytest.mark.parametrize('color', ('red', 'green', 'yellow', 'blue'))
def test_other_rows_keep_original_proposal_scope(monkeypatch, row, column, color):
    node, request = retained(monkeypatch, f'book_col_{column}_row_{row}_{color}')
    assert not mod.verified_bottom_request(node, request)


@pytest.mark.parametrize('model', ('book_col_0_row_5_red', 'book_col_6_row_5_red',
                                  'book_col_01_row_5_red', 'book_col_1_row_15_red'))
def test_noncanonical_names_do_not_enable_bottom(monkeypatch, model):
    node, request = retained(monkeypatch, model)
    assert not mod.verified_bottom_request(node, request)


@pytest.mark.parametrize('flag', mod._RETAINED_FLAGS)
@pytest.mark.parametrize('value', (False, None, 1, 'True'))
def test_unverified_retained_state_does_not_enable_extra_proposals(monkeypatch, flag, value):
    node, request = retained(monkeypatch)
    setattr(node, flag, value)
    assert not mod.verified_bottom_request(node, request)
    with pytest.raises(RuntimeError, match='retained_identity_changed'):
        mod.require_verified_bottom(node, request)


@pytest.mark.parametrize('change', ('target', 'held_copy', 'held_value', 'contact', 'scene', 'hazard', 'cancel'))
def test_actual_request_rejects_changed_context(monkeypatch, change):
    node, request = retained(monkeypatch)
    assert mod.verified_bottom_request(node, request)
    if change == 'target': node._target_book_model = 'book_col_2_row_5_green'
    if change == 'held_copy': node._held_book_corners = node._held_book_corners.copy()
    if change == 'held_value': node._held_book_corners[0, 0] = .001
    if change == 'contact': node._place_contact_guard = PlaceContactGuard(11, vars(request.identity))
    if change == 'scene': node._selected_place_bin_scene['changed'] = True
    if change == 'hazard': node._target_robot_contact_latched = True
    if change == 'cancel': node._cancel.set()
    with pytest.raises(RuntimeError, match='attempt_scene_or_policy_changed'):
        mod.require_verified_bottom(node, request)


def test_capability_cannot_be_substituted(monkeypatch):
    node, request = retained(monkeypatch)
    assert not mod.verified_bottom_request(node, None)
    assert not mod.verified_bottom_request(node, NS(**vars(request)))
    assert not mod.verified_bottom_request(NS(), request)


class Chain:
    lower = np.array([0., *([-3.] * 7)])
    upper = np.array([.4, *([3.] * 7)])
    def __init__(self, radius=1., optimum=-1.8):
        self.radius = radius
        self.optimum = optimum
    def forward(self, q):
        result = np.eye(4)
        result[2, 1] = self.radius * math.cos(q[-1] - self.optimum)
        return result


def inputs():
    first = np.array([.35, 0., 0., 0., 0., 0., 0., -2.1])
    goal = first.copy(); goal[1] = .5; goal[-1] = -1.5
    middle = goal.copy(); middle[1] = .25; middle[-1] = -2.8
    return first, [middle, goal]


def generate(chain, first, goals, checkpoint=lambda: None, **kw):
    return mod.nearest_negative_wrist_goals(chain, first, goals, margin=.1,
        minimum_support=.75, checkpoint=checkpoint, **kw)


def test_nearest_supported_q7_preserves_actual_start_proximal_route_and_exact_endpoint():
    first, goals = inputs(); before = [first.copy(), *(q.copy() for q in goals)]
    chain = Chain()
    result = generate(chain, first, goals)
    assert result[0][-1] == first[-1]
    assert chain.forward(result[0])[2, 1] >= .755
    np.testing.assert_array_equal(result[-1], goals[-1])
    for expected, actual in zip(goals, result):
        np.testing.assert_array_equal(expected[:-1], actual[:-1])
        assert actual[-1] < 0.
        assert not np.shares_memory(expected, actual)
    for original, actual in zip(before, [first, *goals]):
        np.testing.assert_array_equal(original, actual)


def test_nearest_interval_boundary_is_rechecked_with_unrelaxed_reserve():
    first, goals = inputs(); first[-1] = -2.85
    result = generate(Chain(), first, goals)
    assert result[0][-1] > first[-1]
    assert Chain().forward(result[0])[2, 1] >= .755


@pytest.mark.parametrize('chain', (Chain(radius=.754), Chain(optimum=1.8)))
def test_unavailable_negative_supported_interval_is_rejected(chain):
    first, goals = inputs()
    assert generate(chain, first, goals) is None


@pytest.mark.parametrize('where', ('start', 'endpoint', 'intermediate'))
def test_original_joint_limits_remain_mandatory(where):
    first, goals = inputs()
    if where == 'start': first[-1] = .1
    if where == 'endpoint': goals[-1][-1] = .1
    if where == 'intermediate': goals[0][1] = 3.
    assert generate(Chain(), first, goals) is None


@pytest.mark.parametrize('stop_at', (1, 2, 3))
def test_cancellation_or_context_failure_never_returns_a_proposal(stop_at):
    first, goals = inputs(); calls = []
    def check():
        calls.append(None)
        if len(calls) == stop_at: raise RuntimeError('changed context')
    with pytest.raises(RuntimeError, match='changed context'):
        generate(Chain(), first, goals, checkpoint=check)


@pytest.mark.parametrize('bad', ([], [np.zeros(7)], [np.full(8, np.nan)]))
def test_malformed_goals_are_rejected(bad):
    first, _ = inputs()
    with pytest.raises(ValueError): generate(Chain(), first, bad)
