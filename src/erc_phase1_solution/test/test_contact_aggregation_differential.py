"""Exact original/candidate callbacks on real ROS messages, no running nodes.

Every delivery compares the full contact state, immutable histories, peaks,
identity/cancel/latch outcomes and actual hold/status outputs. Operation counts
are algorithm evidence, not callback or planner performance measurements.
"""
from collections import deque
from dataclasses import fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from types import MethodType, SimpleNamespace
import ast
import copy
import hashlib
import math
import random
import threading

import pytest
from ros_gz_interfaces.msg import Contact, Contacts, JointWrench
from erc_phase1_solution import manipulation_node as module
from erc_phase1_solution.place_contact_guard import activate

HERE = Path(__file__).resolve().parent
BASELINE = HERE / 'fixtures/contact_callback_original.py'
CANDIDATE = Path(module.__file__).resolve()
BASELINE_METHOD_AST_SHA256 = 'ca9965b2432b91c8a7be015b5f6f80a91e524e89f1f5067b2477f580a705d012'
MODEL = 'book_col_3_row_2_red'
OTHER_MODEL = 'book_col_2_row_2_red'
BOOK = MODEL + '::book_base_link::base_link_book_collision'
OTHER_BOOK = OTHER_MODEL + '::book_base_link::base_link_book_collision'
BLUE = 'book_col_3_row_2_blue::book_base_link::base_link_book_collision'
LEFT = 'tiago_pro::gripper_left_fingertip_left_link::collision'
INNER = 'tiago_pro::gripper_left_inner_finger_left_link::collision'
RIGHT = 'tiago_pro::gripper_left_fingertip_right_link::collision'
PARKED = 'tiago_pro::gripper_right_fingertip_right_link::collision'
TORSO = 'tiago_pro::torso_lift_link::collision'
ARM = 'tiago_pro::arm_left_3_link::collision'
TABLE = 'erc_table::link::collision'
BIN = 'erc_collection_bin::link::collision'
NOW = 10_000_000_000


def make_message(stamp=NOW, *pairs):
    message = Contacts()
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp, 10**9)
    for first, second, forces in pairs:
        contact = Contact()
        contact.collision1.name, contact.collision2.name = first, second
        if forces is None:
            vectors = []
        elif isinstance(forces, (int, float)):
            vectors = [(float(forces), 0., 0.)]
        else:
            vectors = forces
        for vector in vectors:
            wrench = JointWrench()
            for body, correct in ((wrench.body_1_wrench, 'gripper_left_' in first),
                                  (wrench.body_2_wrench, 'gripper_left_' in second)):
                body.force.x, body.force.y, body.force.z = (
                    tuple(map(float, vector)) if correct else (101., 102., 103.))
            contact.wrenches.append(wrench)
        message.contacts.append(contact)
    return message


@lru_cache(maxsize=2)
def callback_ast(path):
    tree = ast.parse(path.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'ManipulationNode')
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '_on_contacts')
    return method


def callback(path):
    method = callback_ast(path)
    namespace = dict(module.__dict__)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_on_contacts'], namespace


def freeze(value):
    if isinstance(value, float):
        return ('float', value.hex())
    if isinstance(value, dict):
        return tuple((freeze(k), freeze(v)) for k, v in value.items())
    if isinstance(value, (list, tuple, deque)):
        return tuple(map(freeze, value))
    if is_dataclass(value):
        return tuple((f.name, freeze(getattr(value, f.name))) for f in fields(value))
    if hasattr(value, 'get_fields_and_field_types'):
        return tuple((key, freeze(getattr(value, key))) for key in value.get_fields_and_field_types())
    return value


STATE_FIELDS = (
    '_book_contact_samples', '_book_contact_force_samples', '_book_contact_force_frames',
    '_left_target_contact_ns', '_right_target_contact_ns', '_contact_epoch',
    '_contact_generation', '_target_book_model', '_adaptive_force_peaks',
    '_adaptive_overload_latched', '_held_grip_sensor_fault', '_target_robot_contact_latched',
    '_empty_hand_contact', '_last_external_hand_contact_ns', '_empty_arm_contact_latched',
    '_adaptive_motion_halt_reason', '_adaptive_hold_sent', '_place_contact_guard',
    '_held_book_corners', '_transport_lock_engaged', '_adaptive_motion_started',
    '_adaptive_close_active',
)


def make_node(mode='fine', target=MODEL):
    node = object.__new__(module.ManipulationNode)
    now = SimpleNamespace(nanoseconds=NOW)
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    node._lock = threading.Lock()
    node._adaptive_command_lock = threading.RLock()
    node._cancel = threading.Event()
    node.target_colour = 'red'
    node.stock_gripper_close_diagnostic_enabled = mode in ('stock', 'held_stock')
    node._target_book_model = target
    node._book_contact_samples = {}
    node._book_contact_force_samples = {}
    node._book_contact_force_frames = {}
    node._left_target_contact_ns = node._right_target_contact_ns = 0
    node._contact_epoch = node._contact_generation = 0
    node._adaptive_close_active = mode not in ('held', 'held_stock')
    node._adaptive_motion_started = True
    node._adaptive_overload_latched = None
    node._held_grip_sensor_fault = None
    node._target_robot_contact_latched = False
    node._held_book_corners = [(0., 0., 0.)] if mode.startswith('held') else None
    node._transport_lock_engaged = mode.startswith('held')
    node._empty_arm_motion_active = mode == 'empty'
    node.adaptive_contact_force_maximum = 8.
    node.grasp_contact_max_age = .75
    node.joints = {'gripper_left_finger_joint': .017}
    node._latest_gripper_feedback = lambda: module.GripperFeedback(now.nanoseconds, .017, 0., .2)
    node.events, node.published, node.fine_deliveries, node.fine_holds = [], [], [], []
    node._publish_status = lambda *args, **kwargs: node.events.append((args, kwargs))
    node.gripper_pub = SimpleNamespace(publish=lambda msg: node.published.append(copy.deepcopy(msg)))
    node._fine_gripper_controller = SimpleNamespace(
        observe_contacts=lambda msg: node.fine_deliveries.append(freeze(msg)),
        _hold=lambda reason: node.fine_holds.append(reason))
    return node, now


class Pair:
    def __init__(self, mode='fine', target=MODEL):
        self.nodes, self.namespaces = [], []
        self.force_calls, self.sort_calls = [0, 0], [0, 0]
        for index, path in enumerate((BASELINE, CANDIDATE)):
            node, now = make_node(mode, target)
            method, ns = callback(path)
            original_force = module._contact_force_magnitude
            def counted(contact, *, gripper_is_collision1, index=index):
                self.force_calls[index] += 1
                return original_force(contact, gripper_is_collision1=gripper_is_collision1)
            def counted_sort(value, index=index):
                self.sort_calls[index] += 1
                return sorted(value)
            ns['_contact_force_magnitude'] = counted
            ns['sorted'] = counted_sort
            node._on_contacts = MethodType(method, node)
            self.nodes.append(node)
            self.namespaces.append(ns)

    def assert_equal(self):
        def state(node):
            return freeze(dict(
                fields={k: getattr(node, k, '<absent>') for k in STATE_FIELDS},
                cancelled=node._cancel.is_set(), events=node.events, published=node.published,
                # Every prefix was compared on its delivery. Check count and
                # current raw message without quadratically rewalking prefixes.
                fine_deliveries=(len(node.fine_deliveries), node.fine_deliveries[-1:]),
                fine_holds=node.fine_holds))
        assert state(self.nodes[0]) == state(self.nodes[1])

    def deliver(self, message):
        for node in self.nodes:
            node._on_contacts(copy.deepcopy(message))
        self.assert_equal()

    def reset(self, **kwargs):
        for node in self.nodes:
            node._clear_target_contact_samples(**kwargs)
        self.assert_equal()


@pytest.mark.parametrize('mode', ['fine', 'stock', 'held', 'held_stock', 'empty'])
def test_each_delivery_duplicate_pair_partial_fingers_and_out_of_order(mode):
    pair = Pair(mode)
    deliveries = [
        (NOW, (LEFT, BOOK, 3.)),
        (NOW, (BOOK, LEFT, 2.), (RIGHT, BOOK, 4.)),
        (NOW, (INNER, BOOK, 4.), (BOOK, LEFT, 5.)),
        (NOW + 40_000_000, (BOOK, RIGHT, 1.)),
        (NOW + 20_000_000, (LEFT, BOOK, 6.)),
        (NOW, (LEFT, BOOK, 7.)),
        (NOW + 100_000_001, (LEFT, BOOK, 0.), (RIGHT, BOOK, 0.)),
        (NOW - 2_000_000_000, (LEFT, BOOK, 10.)),
        (0, (LEFT, BOOK, 2.)),
        (NOW + 50_000_000,),
    ]
    for stamp, *entries in deliveries:
        pair.deliver(make_message(stamp, *entries))
    assert pair.force_calls[0] == 2 * pair.force_calls[1]
    assert pair.sort_calls[1] > 0  # Out-of-order executes the original rebuild.


def test_pruning_late_peak_reset_and_captured_reader_snapshots():
    pair = Pair('stock')
    for index in range(80):
        pair.deliver(make_message(NOW + index * 1_000_000, (LEFT, BOOK, 1.)))
    node = pair.nodes[1]
    retained = tuple(node._book_contact_force_samples[MODEL][0])
    before = freeze(retained)
    pair.deliver(make_message(NOW, (LEFT, BOOK, 100.)))  # Pruned, but peak remains.
    assert node._adaptive_force_peaks['left']['force_newtons'] == 100.
    assert NOW not in node._book_contact_force_frames[MODEL][0]
    pair.deliver(make_message(NOW + 79_000_000, (INNER, BOOK, 4.)))
    assert freeze(retained) == before
    assert node._book_contact_force_samples[MODEL][0][-1].force_newtons == 5.
    pair.reset(reset_robot_contact=True, reset_target_model=True)
    pair.deliver(make_message(NOW + 79_000_000, (LEFT, BOOK, 2.)))
    assert node._book_contact_force_samples[MODEL][0][-1].force_newtons == 2.
    assert pair.sort_calls[0] == 83 and pair.sort_calls[1] == 1


@pytest.mark.parametrize('bad_force', [None, math.nan, math.inf, -math.inf,
                                     [(1., 2., 3.), (math.nan, 0., 0.)]])
@pytest.mark.parametrize('mode', ['stock', 'held'])
def test_nonfinite_or_empty_wrenches_are_cached_as_invalid_not_recomputed(bad_force, mode):
    pair = Pair(mode)
    pair.deliver(make_message(NOW, (LEFT, BOOK, bad_force), (RIGHT, BOOK, 1.)))
    assert pair.force_calls == [4, 2]
    field = '_adaptive_overload_latched' if mode == 'stock' else '_held_grip_sensor_fault'
    assert getattr(pair.nodes[1], field) == 'invalid_held_contact_force'
    assert pair.nodes[1]._book_contact_force_samples[MODEL][0] == deque(maxlen=64)


@pytest.mark.parametrize('mode', ['stock', 'held_stock'])
def test_nonfinite_aggregate_keeps_latch_and_history(mode):
    pair = Pair(mode)
    # Exercise overflow after individually finite magnitudes. Real norm inputs
    # cannot practically reach this sum; this is the existing aggregate guard.
    for ns in pair.namespaces:
        ns['_contact_force_magnitude'] = lambda *a, **k: 1e308
    pair.deliver(make_message(NOW, (LEFT, BOOK, 1.)))
    pair.deliver(make_message(NOW, (INNER, BOOK, 1.)))
    node = pair.nodes[1]
    assert math.isinf(node._book_contact_force_samples[MODEL][0][-1].force_newtons)
    field = '_adaptive_overload_latched' if mode == 'stock' else '_held_grip_sensor_fault'
    assert getattr(node, field) == 'invalid_stock_contact_force'


@pytest.mark.parametrize('target', [None, MODEL])
@pytest.mark.parametrize('mode', ['fine', 'stock', 'held', 'held_stock'])
def test_identity_multiple_models_non_gripper_robot_and_internal_pairs(target, mode):
    pair = Pair(mode, target)
    for entries in [
        [(PARKED, BOOK, 3.), (LEFT, RIGHT, 2.)],
        [(LEFT, BOOK, 2.), (RIGHT, BOOK, 3.)],
        [(LEFT, OTHER_BOOK, 5.), (RIGHT, BOOK, 1.)],
        [(BOOK, TORSO, 8.)],
        [(BLUE, LEFT, 4.)],
        [(TABLE, INNER, 3.)],
        [],
    ]:
        pair.deliver(make_message(NOW, *entries))


@pytest.mark.parametrize('release', [False, True])
def test_epoch_reset_during_parsing_keeps_original_fault_precedence(release):
    pair = Pair('held_stock')
    for node, ns in zip(pair.nodes, pair.namespaces):
        original = ns['_contact_force_magnitude']
        state = {'done': False}
        def race(*args, node=node, original=original, state=state, **kwargs):
            value = original(*args, **kwargs)
            if not state['done']:
                state['done'] = True
                with node._lock:
                    if release:
                        node._held_book_corners = None
                        node._transport_lock_engaged = False
                    node._clear_target_contact_samples_unlocked(
                        reset_robot_contact=True, reset_target_model=release)
            return value
        ns['_contact_force_magnitude'] = race
    pair.deliver(make_message(NOW, (LEFT, BOOK, 2.), (BOOK, TORSO, 1.)))
    node = pair.nodes[1]
    assert node._book_contact_force_samples == {}
    assert node._target_robot_contact_latched is (not release)


def test_place_contact_and_empty_hand_cancel_holds_are_unchanged():
    pair = Pair('empty')
    for node in pair.nodes:
        node.table_scene_required = True
        activate(node, {'placement_attempt_id': 'attempt-A'})
    pair.deliver(make_message(NOW, (ARM, TABLE, 1.), (LEFT, BOOK, 3.)))
    pair.deliver(make_message(NOW, (BIN, ARM, 2.)))
    pair.deliver(make_message(NOW))
    node = pair.nodes[1]
    assert node._cancel.is_set()
    assert node._place_contact_guard.fault['scene_model'] == 'erc_table'
    assert len(node.published) == 1
    assert [event[0][0] for event in node.events] == [
        'adaptive_gripper_interrupted', 'payload_hazard', 'empty_arm_hazard']


def test_fine_callback_receives_every_original_message_and_error_hold():
    pair = Pair()
    for node in pair.nodes:
        def failed(msg, node=node):
            node.fine_deliveries.append(freeze(msg))
            raise RuntimeError('injected fine observer failure')
        node._fine_gripper_controller.observe_contacts = failed
    pair.deliver(make_message(NOW, (LEFT, BOOK, 2.)))
    assert pair.nodes[1].fine_holds == ['fine_contact_callback_failed']


@pytest.mark.parametrize('bootstrap', ['missing_frames', 'missing_history', 'short_deque'])
def test_inconsistent_bootstrap_uses_original_rebuild(bootstrap):
    pair = Pair()
    pair.deliver(make_message(NOW, (LEFT, BOOK, 1.)))
    for node in pair.nodes:
        if bootstrap == 'missing_frames':
            node._book_contact_force_frames = {}
        elif bootstrap == 'missing_history':
            node._book_contact_force_samples = {}
        else:
            node._book_contact_force_samples[MODEL][0] = deque(
                node._book_contact_force_samples[MODEL][0], maxlen=16)
    pair.deliver(make_message(NOW + 10, (LEFT, BOOK, 2.)))
    assert pair.sort_calls[1] == 1


def test_seeded_stream_compares_state_after_each_of_240_deliveries():
    pair = Pair('stock', None)
    rng = random.Random(370064)
    for index in range(240):
        if index in (90, 170):
            pair.reset(reset_robot_contact=True, reset_target_model=True)
        stamp = NOW + (index if index % 4 else rng.randrange(index + 1)) * 1_000_000
        entries = []
        for _ in range(rng.randrange(4)):
            finger = rng.choice((LEFT, INNER, RIGHT))
            first, second = (finger, BOOK) if rng.randrange(2) else (BOOK, finger)
            entries.append((first, second, rng.choice((0., 1., 3., 5., 8.1))))
        pair.deliver(make_message(stamp, *entries))
    assert pair.force_calls[1] * 2 == pair.force_calls[0]
    assert 0 < pair.sort_calls[1] < pair.sort_calls[0]


def test_original_callback_fixture_is_exact_pinned_method_ast():
    original = ast.dump(callback_ast(BASELINE), include_attributes=False)
    assert hashlib.sha256(original.encode()).hexdigest() == BASELINE_METHOD_AST_SHA256
