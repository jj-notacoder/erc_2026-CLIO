"""Exact finite force results and original malformed-field behavior."""
import ast
from candidate_composition_support import restore_to_r51_source
import hashlib
import math
from pathlib import Path
import random
import struct
from types import SimpleNamespace as NS
from typing import Optional
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE / 'fixtures/contact_norm_original_node.py'
CURRENT = HERE.parent / 'erc_phase1_solution/manipulation_node.py'
PARENT_SHA = '7e04bfb0ef437f8977cbdd6147a97750f7fb816227e0cc72765629245b8abbce'

def load(path):
    tree = ast.parse(path.read_bytes())
    selected = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_contact_force_magnitude')
    scope = {'np': np, 'math': math, 'Optional': Optional}
    exec(compile(ast.Module(body=[selected], type_ignores=[]), str(path), 'exec'), scope)
    return scope['_contact_force_magnitude'], tree

BASELINE, BASE_TREE = load(PARENT)
CANDIDATE, CAND_TREE = load(CURRENT)

def contact(vectors, opposite=(97., 83., 71.)):
    def wrench(v):
        return NS(force=NS(x=v[0], y=v[1], z=v[2]))
    return NS(wrenches=[NS(body_1_wrench=wrench(v),
                          body_2_wrench=wrench(opposite)) for v in vectors])

def outcome(fn, message, side):
    try:
        with np.errstate(all='ignore'):
            value = fn(message, gripper_is_collision1=side)
        return ('none',) if value is None else ('float64', struct.pack('>d', value))
    except Exception as exc:
        return ('exception', type(exc), str(exc))

class ContactNormTests(unittest.TestCase):
    def compare(self, message):
        for side in (True, False):
            self.assertEqual(outcome(BASELINE, message, side),
                             outcome(CANDIDATE, message, side))

    def test_parent_pin_and_only_force_function_changes(self):
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(), PARENT_SHA)
        def remainder(tree):
            return ast.dump(ast.Module(body=[n for n in tree.body if not (
                isinstance(n, ast.FunctionDef) and n.name == '_contact_force_magnitude')],
                type_ignores=[]), include_attributes=False)
        restored = ast.parse(restore_to_r51_source(CURRENT.read_text()))
        self.assertEqual(remainder(BASE_TREE), remainder(restored))

    def test_known_three_four_five_and_opposite_body(self):
        message = contact([(3., 4., 0.)], (0., 0., 12.))
        self.compare(message)
        self.assertEqual(CANDIDATE(message, gripper_is_collision1=True), 5.)
        self.assertEqual(CANDIDATE(message, gripper_is_collision1=False), 12.)

    def test_accumulation_order_and_no_component_cancellation(self):
        for vectors in ([(3., 4., 0.), (-3., -4., 0.)],
                        [(1e15, 0., 0.), (.1, .2, .3)] * 19,
                        [(0., -0., 0.)] * 5):
            self.compare(contact(vectors))
        self.assertEqual(CANDIDATE(contact([(3., 4., 0.), (-3., -4., 0.)]),
                                   gripper_is_collision1=True), 10.)

    def test_exact_finite_random_bit_patterns_and_exponent_edges(self):
        rng = random.Random(43149)
        vectors = []
        for _ in range(2048):
            vectors.append(tuple(struct.unpack('>d', rng.getrandbits(64).to_bytes(8, 'big'))[0]
                                 for _ in range(3)))
        for v in vectors:
            self.compare(contact([v]))
        for i in range(0, len(vectors), 17):
            self.compare(contact(vectors[i:i+17]))

    def test_nonfinite_any_component_and_later_wrench(self):
        for value in (math.nan, math.inf, -math.inf):
            for index in range(3):
                vector = [1., 2., 3.]
                vector[index] = value
                self.compare(contact([tuple(vector)]))
                self.compare(contact([(3., 4., 0.), tuple(vector)]))

    def test_underflow_overflow_and_threshold_neighbors(self):
        values = [0., -0., 5e-324, -5e-324, 1e-200, 1e200,
                  float(np.finfo(float).max)]
        for threshold in (.001, .5, 3.5, 8., 10., 40.):
            values.extend((float(np.nextafter(threshold, -math.inf)), threshold,
                           float(np.nextafter(threshold, math.inf))))
        for value in values:
            for vector in ((value, 0., 0.), (value, value, value)):
                self.compare(contact([vector]))

    def test_empty_and_missing_fields(self):
        for message in (NS(), NS(wrenches=[]), NS(wrenches=[NS()]),
                        NS(wrenches=[NS(body_1_wrench=NS(force=NS(x=1., y=2.))) ])):
            self.compare(message)

    def test_original_conversion_and_nonstandard_shape(self):
        for vector in ((1, 2, 3), (True, False, True), ('3', '4', '0'),
                       ('bad', 1., 2.), (None, 1., 2.),
                       ([1., 2.], [3., 4.], [5., 6.]),
                       ([1.], [2.], [3.]),
                       ([1., 2.], [3.], [4.]),
                       (complex(1., 2.), 0., 0.)):
            self.compare(contact([vector]))

if __name__ == '__main__':
    unittest.main()
