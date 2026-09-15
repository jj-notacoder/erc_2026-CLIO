"""Continuation must complete a real checked path without relaxing its gates."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import test_scene_cartesian_solver as base

m = base.m


class CompletionTests(unittest.TestCase):
    run_solver = base.SolverTests.run_solver
    validator = base.SolverTests.validator

    def setUp(self):
        base.SolverTests.setUp(self)
        self.positions = [[.1+.02*i, 0., .2] for i in range(11)]
        # Model the actual motivating case: loose whole-path strategy fails,
        # but the tight first-waypoint frontier contains useful live roots.
        self.chain.modifier = lambda q, *_: (
            None if self.chain.calls[-1][2]['position_tolerance'] == .012 else q)

    def test_eleven_waypoints_complete_before_widening_frontier(self):
        info = {}
        path, *_ = self.run_solver(limits=m.SearchLimits(max_ik_calls=80), diagnostics_out=info)
        self.assertEqual(len(path), 11)
        self.assertTrue(info['completion_first']['accepted'])
        self.assertEqual(info['completion_first']['completed_paths'], 1)
        self.assertEqual(info['completion_first']['ik_calls'], 10)
        self.assertEqual(info['strategy'], {'loose_paths': 0, 'refined_paths': 0})
        self.assertEqual(len(self.validations), 1)
        self.assertTrue(any(aperture == .019 for _, aperture, _ in self.scene.calls))
        np.testing.assert_allclose([self.chain.forward(q)[:3, 3] for q in path], self.positions)

    def test_completed_last_allowed_solve_still_gets_full_admission(self):
        info = {}
        path, *_ = self.run_solver(limits=m.SearchLimits(max_ik_calls=44), diagnostics_out=info)
        self.assertEqual(info['completion_first']['ik_budget'], 10)
        self.assertEqual(info['completion_first']['ik_calls'], 10)
        self.assertTrue(info['completion_first']['accepted'])
        self.assertEqual(len(path), 11)
        self.assertEqual(len(self.validations), 1)

    def test_local_cap_preserves_remaining_global_budget_and_fallback(self):
        with self.assertRaisesRegex(m.SceneCartesianSearchError, 'ik_budget') as caught:
            self.run_solver(limits=m.SearchLimits(max_ik_calls=26))
        detail = caught.exception.diagnostics
        self.assertEqual(detail['completion_first']['ik_budget'], 1)
        self.assertEqual(detail['completion_first']['ik_calls'], 1)
        self.assertFalse(detail['completion_first']['accepted'])
        self.assertEqual(detail['ik_calls'], 26)
        self.assertEqual(len(self.chain.calls), 26)
        self.assertEqual(self.validations, [])

    def test_full_validator_rejection_cannot_be_turned_into_success(self):
        self.validator = lambda path, transition: self.validations.append(path) or False
        with self.assertRaisesRegex(m.SceneCartesianSearchError, 'candidate_budget'):
            self.run_solver(limits=m.SearchLimits(max_ik_calls=160, max_candidates=2))
        self.assertEqual(len(self.validations), 2)
        self.assertTrue(all(len(path) == 11 for path in self.validations))

    def test_opening_failure_never_calls_full_validator(self):
        self.scene.accept = lambda q, a, loaded: a < .019
        with self.assertRaises(m.SceneCartesianSearchError):
            self.run_solver(limits=m.SearchLimits(max_ik_calls=100))
        self.assertTrue(any(a == .019 for _, a, _ in self.scene.calls))
        self.assertEqual(self.validations, [])

    def test_original_beam_can_recover_after_greedy_dead_end(self):
        self.positions = [[.1, 0., .2], [.2, 0., .2], [.3, 0., .2]]
        def accept(q, a, loaded):
            allowed = (-1.,) if q[1] < .15 else (-1., -1.125) if q[1] < .25 else (-1.5,)
            return any(abs(q[3]-v) < 1e-8 for v in allowed)
        self.scene.accept = accept
        info = {}
        path, *_ = self.run_solver(diagnostics_out=info)
        self.assertFalse(info['completion_first']['accepted'])
        self.assertEqual(info['completion_first']['completed_paths'], 0)
        np.testing.assert_allclose([q[3] for q in path], [-1., -1.125, -1.5])

    def test_cancel_during_completion_endpoint_cannot_reach_validator(self):
        def accept(q, a, loaded):
            if q[1] > .11: self.node._cancel.set()
            return True
        self.scene.accept = accept
        with self.assertRaisesRegex(m.SceneCartesianSearchError, 'cancelled') as caught:
            self.run_solver()
        self.assertGreater(caught.exception.diagnostics['completion_first']['ik_calls'], 0)
        self.assertEqual(self.validations, [])

    def test_cancel_during_complete_validator_discards_accepted_path(self):
        self.validator = lambda *_: self.node._cancel.set() or True
        with self.assertRaisesRegex(m.SceneCartesianSearchError, 'cancelled') as caught:
            self.run_solver()
        self.assertEqual(caught.exception.diagnostics['completion_first']['completed_paths'], 1)

    def test_original_wall_deadline_includes_completion_phase(self):
        clock = [0.]
        def modify(q, *_):
            clock[0] += .01
            return None if self.chain.calls[-1][2]['position_tolerance'] == .012 else q
        self.chain.modifier = modify
        with patch.object(m.time, 'monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(m.SceneCartesianSearchError, 'wall_budget') as caught:
                self.run_solver(limits=m.SearchLimits(wall_seconds=.28))
        self.assertGreater(caught.exception.diagnostics['completion_first']['ik_calls'], 0)
        self.assertLessEqual(len(self.chain.calls), 28)
        self.assertEqual(self.validations, [])

    def test_full_validator_still_receives_owned_copies(self):
        def modify(path, transition):
            path[-1][:] = 999.
            return True
        self.validator = modify
        path, *_ = self.run_solver()
        self.assertTrue(np.all(path[-1] < 3.))

    def test_joint_step_guard_still_blocks_completion(self):
        self.positions[1] = [.501, 0., .2]
        with self.assertRaises(m.SceneCartesianSearchError):
            self.run_solver(limits=m.SearchLimits(max_ik_calls=100))
        self.assertEqual(self.validations, [])

    def test_orientation_residual_guard_still_blocks_completion(self):
        original = self.chain.pose_error
        def error(actual, target):
            result = original(actual, target)
            if target[0, 3] > .11: result[3] = .020001
            return result
        self.chain.pose_error = error
        with self.assertRaises(m.SceneCartesianSearchError):
            self.run_solver(limits=m.SearchLimits(max_ik_calls=100))
        self.assertEqual(self.validations, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
