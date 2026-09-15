"""Bounded numerical equivalence tests; run only in an assigned test lane.

Tests use the actual package module and required official sibling URDF.
The reference is the unchanged public forward method, including a subclass
that selects the ordinary solver loop rather than the optimized context.
"""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np

from erc_phase1_solution import kinematics as mod
URDF = Path(__file__).resolve().parents[2]/'erc_description/urdf/tiago_pro.urdf'
Chain, Joint = mod.URDFChain, mod.Joint
METRICS = dict(columns=0, maximum_pose_entry_error=0., maximum_jacobian_entry_error=0.)


class OrdinaryChain(Chain):
    """Same forward semantics, but explicitly unsupported for factor reuse."""


def make_chain(rng, count=9, subset=False, reverse=False):
    joints = []
    movable = []
    for i in range(count):
        kind = ('fixed', 'revolute', 'prismatic', 'continuous')[i % 4]
        axis = rng.normal(size=3)
        # Exercise the exact original zero-axis rotation convention as well.
        if i == 5:
            axis[:] = 0.
        if kind == 'prismatic':
            axis *= 1.7  # The original prismatic convention does not normalize.
        origin = mod._transform(mod._rpy_matrix(rng.uniform(-1., 1., 3)), rng.uniform(-.2, .2, 3))
        lower, upper = (-.15, .25) if kind == 'prismatic' else (-2.2, 2.5)
        if kind == 'fixed':
            lower = upper = 0.
        joint = Joint(f'j{i}', f'l{i}', f'l{i+1}', kind, origin, axis, lower, upper)
        joints.append(joint)
        if kind != 'fixed':
            movable.append(joint.name)
    selected = movable[::2] if subset else movable
    if reverse:
        selected = selected[::-1]
    return Chain(joints, selected)


def check_columns(case, chain, q, fixed=(), lower=None, upper=None):
    q = np.asarray(q, dtype=float)
    lower = chain.lower if lower is None else lower
    upper = chain.upper if upper is None else upper
    current = chain.forward(q)
    varied = chain._forward_variations(q)
    case.assertIsNotNone(varied)
    saved = q.copy()
    for i in range(len(q)):
        if i in fixed:
            continue
        trial = q.copy()
        trial[i] = min(upper[i], trial[i] + 1e-5)
        step = trial[i] - q[i]
        if abs(step) < 1e-10:
            trial[i] = max(lower[i], q[i] - 1e-5)
            step = trial[i] - q[i]
        if abs(step) < 1e-10:
            continue
        expected = chain.forward(trial)
        actual = varied(i, float(trial[i]))
        pose_difference = float(np.max(np.abs(actual-expected)))
        expected_column = chain.pose_error(current, expected)/step
        actual_column = chain.pose_error(current, actual)/step
        jac_difference = float(np.max(np.abs(actual_column-expected_column)))
        METRICS['columns'] += 1
        METRICS['maximum_pose_entry_error'] = max(METRICS['maximum_pose_entry_error'], pose_difference)
        METRICS['maximum_jacobian_entry_error'] = max(METRICS['maximum_jacobian_entry_error'], jac_difference)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_column, expected_column)
    np.testing.assert_array_equal(saved, q)


class ForwardVariationTests(unittest.TestCase):
    def test_random_mixed_axes_fixed_links_active_subsets_and_order(self):
        rng = np.random.default_rng(915623)
        for count in (4, 7, 10, 13):
            for subset in (False, True):
                for reverse in (False, True):
                    chain = make_chain(rng, count, subset, reverse)
                    for _ in range(8):
                        check_columns(self, chain, rng.uniform(chain.lower, chain.upper))

    def test_limits_backward_steps_narrow_ranges_and_fixed_solver_coordinates(self):
        chain = make_chain(np.random.default_rng(106), 10)
        for q in (chain.lower, chain.upper, chain.upper-3e-6):
            check_columns(self, chain, q)
        q = (chain.lower+chain.upper)/2
        lower, upper = q-2e-6, q+2e-6
        lower[0] = upper[0] = q[0]
        check_columns(self, chain, q, fixed=(1, 3), lower=lower, upper=upper)

    def test_official_full_intermediate_and_selected_joint_chains(self):
        self.assertTrue(URDF.is_file(), f'Required pinned official URDF missing: {URDF}')
        rng = np.random.default_rng(3182026)
        pairs = [('base_footprint', 'gripper_left_grasping_link'),
                 ('base_footprint', 'arm_left_5_link'),
                 ('torso_lift_link', 'arm_left_7_link'),
                 ('arm_left_3_link', 'gripper_left_grasping_link'),
                 ('base_footprint', 'gripper_right_grasping_link')]
        for base, tip in pairs:
            full = Chain.from_urdf(URDF, base, tip)
            for names in (full.active_names, full.active_names[::2][::-1]):
                chain = Chain.from_urdf(URDF, base, tip, names)
                for q in (chain.lower, chain.upper, *[rng.uniform(chain.lower, chain.upper) for _ in range(6)]):
                    check_columns(self, chain, q)

    def test_subclass_and_instance_forward_override_fall_back(self):
        chain = make_chain(np.random.default_rng(20))
        subclass = OrdinaryChain(chain.joints, chain.active_names)
        self.assertIsNone(subclass._forward_variations(np.zeros(len(chain.active_names))))
        original = chain.forward
        chain.forward = lambda positions: original(positions)
        self.assertIsNone(chain._forward_variations(np.zeros(len(chain.active_names))))

    def test_ambiguous_or_custom_joint_layouts_fall_back(self):
        rng = np.random.default_rng(63)
        examples = []
        duplicate = make_chain(rng)
        duplicate.active_names.append(duplicate.active_names[0])
        examples.append(duplicate)
        duplicate_joint = make_chain(rng)
        duplicate_joint.joints.append(duplicate_joint.joints[-1])
        examples.append(duplicate_joint)
        mismatched_index = make_chain(rng)
        mismatched_index._active_index[mismatched_index.active_names[0]] = 99
        examples.append(mismatched_index)
        unsupported = make_chain(rng)
        j = unsupported.joints[0]
        unsupported.joints[0] = Joint(j.name, j.parent, j.child, 'floating', j.origin, j.axis, j.lower, j.upper)
        examples.append(unsupported)
        custom = make_chain(rng)
        class CustomJoint(Joint):
            pass
        j = custom.joints[0]
        custom.joints[0] = CustomJoint(j.name, j.parent, j.child, j.kind, j.origin, j.axis, j.lower, j.upper)
        examples.append(custom)
        for chain in examples:
            self.assertIsNone(chain._forward_variations(np.zeros(len(chain.active_names))))

    def test_solver_success_and_fixed_joint_results_match_ordinary_path(self):
        chain = Chain.from_urdf(URDF, 'torso_lift_link', 'gripper_left_grasping_link')
        ordinary = OrdinaryChain(chain.joints, chain.active_names)
        rng = np.random.default_rng(208711)
        for fixed in ({}, {chain.active_names[2]: float((chain.lower[2]+chain.upper[2])/2)}):
            target_q = (chain.lower+chain.upper)/2
            for name, value in fixed.items():
                target_q[chain._active_index[name]] = value
            target = chain.forward(target_q)
            seed = target_q+rng.uniform(-.04, .04, len(target_q))
            kwargs = dict(max_iterations=35, position_tolerance=1e-5, orientation_tolerance=1e-4, fixed_positions=fixed)
            expected, score = ordinary.solve(target, [seed], **kwargs)
            actual, actual_score = chain.solve(target, [seed], **kwargs)
            self.assertIsNotNone(expected)
            self.assertIsNotNone(actual)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual_score, score)
            for name, value in fixed.items():
                self.assertEqual(actual[chain._active_index[name]], value)

    def test_solver_failure_and_single_free_joint_use_existing_semantics(self):
        chain = make_chain(np.random.default_rng(16), count=4)
        ordinary = OrdinaryChain(chain.joints, chain.active_names)
        target = np.eye(4)
        target[:3, 3] = [100., 80., 90.]
        seed = (chain.lower+chain.upper)/2
        for fixed in ({}, {name: float(seed[i]) for i, name in enumerate(chain.active_names) if i != 0}):
            expected, score = ordinary.solve(target, [seed], max_iterations=3, fixed_positions=fixed)
            actual, actual_score = chain.solve(target, [seed], max_iterations=3, fixed_positions=fixed)
            self.assertIsNone(expected)
            self.assertIsNone(actual)
            self.assertEqual(actual_score, score)

    def test_target_error_trajectory_near_zero_and_pi_branches(self):
        chain = Chain.from_urdf(URDF, 'torso_lift_link', 'gripper_left_grasping_link')
        ordinary = OrdinaryChain(chain.joints, chain.active_names)
        seed = (chain.lower+chain.upper)/2
        initial = chain.forward(seed)
        original_error = Chain.pose_error
        angles = (0., 1e-9, .999e-7, 1.001e-7, 1e-5,
                  math.pi-1.001e-5, math.pi-.999e-5, math.pi)
        for angle in angles:
            target = initial.copy()
            target[:3, :3] = mod._axis_rotation([.7, -.2, .5], angle) @ initial[:3, :3]
            traces = []
            outcomes = []
            for tested in (ordinary, chain):
                samples = []
                def tracking_error(current, goal):
                    value = original_error(current, goal)
                    if goal is target:
                        samples.append((current.copy(), value.copy()))
                    return value
                Chain.pose_error = staticmethod(tracking_error)
                try:
                    outcomes.append(tested.solve(target, [seed], max_iterations=12,
                                                 position_tolerance=1e-10, orientation_tolerance=1e-10))
                finally:
                    Chain.pose_error = staticmethod(original_error)
                traces.append(samples)
            # The initial nominal forward/error must be exactly untouched.
            np.testing.assert_array_equal(traces[0][0][0], traces[1][0][0])
            np.testing.assert_array_equal(traces[0][0][1], traces[1][0][1])
            self.assertEqual(len(traces[0]), len(traces[1]), f'iteration count at angle {angle}')
            for (pose, error), (actual_pose, actual_error) in zip(*traces):
                pose_delta = float(np.max(np.abs(pose-actual_pose)))
                error_delta = float(np.max(np.abs(error-actual_error)))
                METRICS['maximum_target_trace_pose_error'] = max(METRICS.get('maximum_target_trace_pose_error', 0.), pose_delta)
                METRICS['maximum_target_trace_error_vector_difference'] = max(METRICS.get('maximum_target_trace_error_vector_difference', 0.), error_delta)
                np.testing.assert_array_equal(actual_pose, pose, err_msg=f'angle={angle}')
                np.testing.assert_array_equal(actual_error, error, err_msg=f'angle={angle}')
            self.assertEqual(outcomes[0][0] is None, outcomes[1][0] is None)
            self.assertEqual(outcomes[0][1], outcomes[1][1])


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ForwardVariationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(METRICS, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
