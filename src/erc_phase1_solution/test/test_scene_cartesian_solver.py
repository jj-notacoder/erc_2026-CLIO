"""Portable behavioral tests: fake kinematics/scene, actual bounded solver."""
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]/'erc_phase1_solution'/'scene_cartesian_solver.py'
spec = importlib.util.spec_from_file_location('scene_cartesian_solver_under_test', SOURCE)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
ROTATION = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


class Chain:
    active_names = ['torso_lift_joint']+[f'arm_left_{i}_joint' for i in range(1, 8)]
    lower = np.array([0., -3., -3., -3., -3., -3., -3., -3.])
    upper = np.array([.4, 3., 3., 3., 3., 3., 3., 3.])

    def __init__(self):
        self.calls = []
        self.modifier = None
        self.support = 1.

    def solve(self, matrix, seeds, **kwargs):
        self.calls.append((matrix.copy(), [s.copy() for s in seeds], dict(kwargs)))
        fixed = kwargs['fixed_positions']
        q = seeds[0].copy()
        q[0] = fixed['torso_lift_joint']
        q[1], q[2], q[4] = matrix[:3, 3]
        q[3] = fixed.get('arm_left_3_joint', -1.)
        q[5], q[6], q[7] = q[3]/10., 0., -1.
        if self.modifier is not None:q = self.modifier(q, matrix, fixed)
        return q, 0.

    def forward(self, q):
        result = np.eye(4)
        result[:3, :3] = ROTATION
        result[2, 1] = self.support
        result[:3, 3] = [q[1], q[2], q[4]]
        return result

    @staticmethod
    def pose_error(actual, target):
        # The fixture varies only translation; support is independently varied
        # for the signed-support guard without inventing an IK angular error.
        return np.r_[target[:3, 3]-actual[:3, 3], [0., 0., 0.]]


class Scene:
    def __init__(self):
        self.calls = []
        self.accept = lambda q, aperture, loaded:True

    def sample(self, q, aperture, loaded):
        self.calls.append((q.copy(), aperture, loaded))
        return self.accept(q, aperture, loaded)


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.chain = Chain()
        self.node = SimpleNamespace(chain=self.chain, _cancel=threading.Event(),
                                    place_joint_limit_margin=.1,
                                    carried_supported_jaw_vertical_component=.75)
        self.scene = Scene()
        self.seed = np.array([.3, .1, 0., -1., .2, 0., 0., -1.])
        self.positions = [[.1, 0., .2], [.2, 0., .2], [.3, 0., .2]]
        self.validations = []

    def validator(self, path, transition):
        self.validations.append(([q.copy() for q in path], transition))
        return True

    def run_solver(self, **kwargs):
        settings = dict(measured_seed=self.seed, aperture=.017, open_aperture=.019,
                        limits=m.SearchLimits(max_ik_calls=512, max_paths=4))
        settings.update(kwargs)
        return m.solve_scene_cartesian(self.node, self.positions, ROTATION, .3,
                                        self.seed.copy(), self.scene, self.validator, **settings)

    def test_live_targets_and_supplied_seed_are_used_without_mutation(self):
        original = self.seed.copy()
        self.positions = [[.33, .02, .21], [.35, .02, .21]]
        info = {}
        path, orientation, score, transition = self.run_solver(diagnostics_out=info)
        self.assertEqual(orientation, 0)
        self.assertEqual(transition, [])
        self.assertGreaterEqual(score, 0.)
        np.testing.assert_allclose(self.chain.forward(path[-1])[:3, 3], self.positions[-1])
        np.testing.assert_array_equal(self.seed, original)
        np.testing.assert_array_equal(self.chain.calls[0][1][0], original)
        self.assertTrue(self.validations)
        self.assertFalse(info['recorded_joint_path_used'])
        for _, _, values in self.chain.calls:
            self.assertIn((values['position_tolerance'], values['orientation_tolerance']),
                          ((.002, .02), (.012, .10)))
            self.assertEqual(values['joint_limit_margin'], .1)
        self.assertEqual(info['position_tolerance'], .002)

    def test_variable_swivel_avoids_an_endpoint_obstacle(self):
        self.scene.accept = lambda q, aperture, loaded: (
            abs(q[3]-(-1. if q[1] < .15 else -1.125 if q[1] < .25 else -1.25)) < 1e-8)
        path, *_ = self.run_solver()
        np.testing.assert_allclose([q[3] for q in path], [-1., -1.125, -1.25])
        self.assertTrue(all(np.max(abs(b[1:]-a[1:])) <= .4 for a, b in zip(path, path[1:])))

    def test_loose_branch_is_refined_from_each_own_waypoint_seed(self):
        loose_targets = []
        refined_targets = []
        def modify(q, matrix, fixed):
            _, seeds, options = self.chain.calls[-1]
            if options['position_tolerance'] == .012:
                loose_targets.append(matrix[:3, 3].copy())
                q[6] = .2
                q[1] += .005  # Valid loose seed, unacceptable final error.
                return q
            if seeds[0][6] != .2:return None
            refined_targets.append(matrix[:3, 3].copy())
            q[6] = .2
            return q
        self.chain.modifier = modify
        info = {}
        path, *_ = self.run_solver(diagnostics_out=info)
        np.testing.assert_allclose(loose_targets, self.positions)
        np.testing.assert_allclose(refined_targets, self.positions)
        np.testing.assert_allclose([self.chain.forward(q)[:3, 3] for q in path], self.positions)
        self.assertEqual(info['strategy'], {'loose_paths':1, 'refined_paths':1})

    def test_failed_tight_refinement_cannot_accept_loose_pose(self):
        def modify(q, matrix, fixed):
            return q if self.chain.calls[-1][2]['position_tolerance'] == .012 else None
        self.chain.modifier = modify
        with self.assertRaisesRegex(m.SceneCartesianSearchError, 'no_accepted_route'):self.run_solver()
        self.assertEqual(self.validations, [])

    def test_endpoint_rejection_never_reaches_full_validator(self):
        self.scene.accept = lambda q, aperture, loaded:False
        with self.assertRaisesRegex(m.SceneCartesianSearchError, 'no_accepted_route'):
            self.run_solver()
        self.assertEqual(self.validations, [])

    def test_opening_rejection_never_reaches_full_validator(self):
        self.scene.accept = lambda q, aperture, loaded:aperture <= .017
        with self.assertRaisesRegex(m.SceneCartesianSearchError, 'no_accepted_route'):
            self.run_solver()
        self.assertTrue(any(a > .017 for _, a, _ in self.scene.calls))
        self.assertEqual(self.validations, [])

    def test_opening_rejection_tries_another_path(self):
        self.positions = self.positions[:1]
        self.scene.accept = lambda q, a, loaded:a <= .017 or abs(q[3]+1.) > 1e-8
        path, *_ = self.run_solver()
        self.assertNotAlmostEqual(path[-1][3], -1.)
        self.assertTrue(any(a > .017 and abs(q[3]+1.) < 1e-8 for q, a, _ in self.scene.calls))

    def test_full_candidate_rejection_tries_next_retained_path(self):
        self.positions = self.positions[:1]
        seen = []
        def validate(path, transition):
            seen.append(path[-1][3])
            return len(seen) == 2
        self.validator = validate
        path, *_ = self.run_solver()
        self.assertEqual(len(seen), 2)
        self.assertNotEqual(seen[0], seen[1])
        self.assertEqual(path[-1][3], seen[1])

    def test_full_rejection_does_not_return_an_endpoint_only_plan(self):
        self.validator = lambda path, transition:False
        with self.assertRaises(m.SceneCartesianSearchError):self.run_solver()

    def test_validator_cannot_mutate_returned_solution(self):
        def validate(path, transition):
            path[-1][:] = 999.
            return True
        self.validator = validate
        path, *_ = self.run_solver()
        self.assertTrue(np.all(path[-1] < 3.))

    def test_cancel_before_search_runs_no_ik_or_scene(self):
        self.node._cancel.set()
        with self.assertRaisesRegex(m.SceneCartesianSearchError, 'cancelled'):self.run_solver()
        self.assertEqual(self.chain.calls, [])
        self.assertEqual(self.scene.calls, [])

    def test_cancel_during_ik_stops_before_scene(self):
        def modifier(q, *_):
            self.node._cancel.set()
            return q
        self.chain.modifier = modifier
        with self.assertRaisesRegex(m.SceneCartesianSearchError, 'cancelled'):self.run_solver()
        self.assertEqual(len(self.chain.calls), 1)
        self.assertEqual(self.scene.calls, [])

    def test_cancel_during_scene_stops_before_validator(self):
        def accept(*_):
            self.node._cancel.set()
            return True
        self.scene.accept = accept
        with self.assertRaisesRegex(m.SceneCartesianSearchError, 'cancelled'):self.run_solver()
        self.assertEqual(self.validations, [])

    def test_cancel_during_opening_stops_before_validator(self):
        def accept(q, a, loaded):
            if a > .017:self.node._cancel.set()
            return True
        self.scene.accept = accept
        with self.assertRaisesRegex(m.SceneCartesianSearchError, 'cancelled'):self.run_solver()
        self.assertEqual(self.validations, [])

    def test_cancel_inside_successful_full_validator_cannot_return(self):
        def validate(*_):
            self.node._cancel.set()
            return True
        self.validator = validate
        with self.assertRaisesRegex(m.SceneCartesianSearchError, 'cancelled'):self.run_solver()

    def test_joint_step_above_point_four_rejected(self):
        self.positions = [[.1, 0., .2], [.501, 0., .2]]
        with self.assertRaisesRegex(m.SceneCartesianSearchError, 'no_accepted_route'):self.run_solver()
        self.assertEqual(self.validations, [])

    def test_margin_and_fixed_torso_are_enforced_on_solver_output(self):
        for index, value in ((1, 2.901), (0, .31)):
            with self.subTest(index=index):
                def modify(q, *_, index=index, value=value):
                    q[index] = value
                    return q
                self.chain.modifier = modify
                with self.assertRaisesRegex(m.SceneCartesianSearchError, 'no_accepted_route'):self.run_solver()
        self.assertEqual(self.validations, [])

    def test_negative_wrist_and_signed_support_required(self):
        self.chain.modifier = lambda q, *_:np.r_[q[:-1], 1.]
        with self.assertRaisesRegex(m.SceneCartesianSearchError, 'no_accepted_route'):self.run_solver()
        self.chain.modifier = None
        self.chain.support = .74999
        with self.assertRaisesRegex(m.SceneCartesianSearchError, 'no_accepted_route'):self.run_solver()
        self.assertEqual(self.validations, [])

    def test_wrong_pose_cannot_be_accepted_from_ik_success(self):
        self.chain.modifier = lambda q, *_:q+np.array([0., .003, 0., 0., 0., 0., 0., 0.])
        with self.assertRaisesRegex(m.SceneCartesianSearchError, 'no_accepted_route'):self.run_solver()
        self.assertEqual(self.validations, [])

    def test_ik_budget_is_shared_across_branches(self):
        with self.assertRaisesRegex(m.SceneCartesianSearchError, 'ik_budget'):
            self.run_solver(limits=m.SearchLimits(max_ik_calls=3))
        self.assertEqual(len(self.chain.calls), 3)

    def test_wall_deadline_does_not_reset_across_ik(self):
        clock = [0.]
        def modify(q, *_):
            clock[0] += .4
            return q
        self.chain.modifier = modify
        with patch.object(m.time, 'monotonic', side_effect=lambda:clock[0]):
            with self.assertRaisesRegex(m.SceneCartesianSearchError, 'wall_budget'):
                self.run_solver(limits=m.SearchLimits(wall_seconds=1.))
        self.assertEqual(len(self.chain.calls), 3)

    def test_candidate_budget_stops_further_full_validations(self):
        seen = []
        self.positions = self.positions[:1]
        self.validator = lambda path, transition:seen.append(path) or False
        with self.assertRaisesRegex(m.SceneCartesianSearchError, 'candidate_budget'):
            self.run_solver(limits=m.SearchLimits(max_candidates=1))
        self.assertEqual(len(seen), 1)

    def test_successful_validator_cannot_erase_original_wall_deadline(self):
        clock = [0.]
        def validate(*_):
            clock[0] = 2.
            return True
        self.validator = validate
        with patch.object(m.time, 'monotonic', side_effect=lambda:clock[0]):
            with self.assertRaisesRegex(m.SceneCartesianSearchError, 'wall_budget'):
                self.run_solver(limits=m.SearchLimits(wall_seconds=1.))

    def test_nonfinite_configured_guards_are_not_silently_defaulted(self):
        for name in ('place_joint_limit_margin', 'carried_supported_jaw_vertical_component'):
            with self.subTest(name=name):
                old = getattr(self.node, name)
                setattr(self.node, name, float('nan'))
                with self.assertRaises(ValueError):self.run_solver()
                setattr(self.node, name, old)
        self.assertEqual(self.chain.calls, [])

    def test_invalid_inputs_cannot_start_search(self):
        for kwargs in ({'open_aperture':.069001}, {'aperture':float('nan')},
                       {'measured_seed':[0.]}, {'limits':m.SearchLimits(max_paths=0)}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):self.run_solver(**kwargs)
        self.assertEqual(self.chain.calls, [])

    def test_no_full_validator_is_rejected(self):
        self.validator = None
        with self.assertRaisesRegex(ValueError, 'validator'):self.run_solver()
        self.assertEqual(self.chain.calls, [])


if __name__ == '__main__':unittest.main(verbosity=2)
