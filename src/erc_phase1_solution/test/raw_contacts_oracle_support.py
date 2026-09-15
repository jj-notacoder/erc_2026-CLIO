"""Independent installed-IDL oracle and original R52 callback fixtures.

No node construction, ROS init, subscriptions, executor or geometry. Import only
in the coordinated validation process with sourced installed ROS dependencies.
"""
import array
import ast
from collections import deque
import hashlib
import importlib.util
import os
from pathlib import Path
import random
import struct
import sys

from ros_gz_interfaces.msg import Contact, Contacts, JointWrench
from geometry_msgs.msg import Vector3
from rclpy.serialization import serialize_message, deserialize_message

BASE = Path(__file__).resolve().parents[1]
from erc_phase1_solution import raw_contacts as raw
assert Path(raw.__file__).resolve() == BASE / 'erc_phase1_solution/raw_contacts.py'
# Explicit universal-newline normalization; source pin is Python-version independent.
_node_text = (BASE / 'erc_phase1_solution/manipulation_node.py').read_text().replace('\r\n', '\n').replace('\r', '\n')
_tree = ast.parse(_node_text)
_cls = next(n for n in _tree.body if isinstance(n, ast.ClassDef) and n.name == 'ManipulationNode')
_method = next(n for n in _cls.body if isinstance(n, ast.FunctionDef) and n.name == '_on_contacts')
assert hashlib.sha256(ast.get_source_segment(_node_text, _method).encode()).hexdigest() == 'bcc72b9d3f04f37846c9a217dd7fc4aa1cfbfeec35422f26c949874951eed94c'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

fixture = load('independent_r52_contact_fixture', BASE / 'test/test_contact_aggregation_differential.py')
assert Path(fixture.module.__file__).resolve() == BASE / 'erc_phase1_solution/manipulation_node.py'

# Explicit complete schema from the copied installed IDL, not decoder metadata.
SCHEMA = {
    'Time': [('sec', 'i'), ('nanosec', 'I')],
    'Header': [('stamp', 'Time'), ('frame_id', 'string')],
    'Entity': [('id', 'Q'), ('name', 'string'), ('type', 'B')],
    'Vector3': [('x', 'd'), ('y', 'd'), ('z', 'd')],
    'Wrench': [('force', 'Vector3'), ('torque', 'Vector3')],
    'String': [('data', 'string')],
    'UInt32': [('data', 'I')],
    'JointWrench': [('header', 'Header'), ('body_1_name', 'String'),
                   ('body_1_id', 'UInt32'), ('body_2_name', 'String'),
                   ('body_2_id', 'UInt32'), ('body_1_wrench', 'Wrench'),
                   ('body_2_wrench', 'Wrench')],
    'Contact': [('collision1', 'Entity'), ('collision2', 'Entity'),
                ('positions', ('seq', 'Vector3')), ('normals', ('seq', 'Vector3')),
                ('depths', ('seq', 'd')), ('wrenches', ('seq', 'JointWrench'))],
    'Contacts': [('header', 'Header'), ('contacts', ('seq', 'Contact'))],
}

def canonical(value, kind='Contacts'):
    if isinstance(kind, tuple):
        return tuple(canonical(item, kind[1]) for item in value)
    if kind in SCHEMA:
        return tuple((field, canonical(getattr(value, field), typ)) for field, typ in SCHEMA[kind])
    if kind == 'd':
        # Exact IEEE bits, including signed zero and NaN payloads.
        return ('f64', struct.pack('>d', value).hex())
    return value

class IDLEncoder:
    """Test-only opposite-endian writer, independently checked by stock ROS.

    Field-offset annotations also support targeted malformed mutations. This is
    not a second production decoder and is never used to establish an oracle
    without comparing stock deserialize_message first.
    """
    def __init__(self, endian='<'):
        self.endian = endian
        self.buf = bytearray(b'\x00\x01\x00\x00' if endian == '<' else b'\x00\x00\x00\x00')
        self.offsets = []

    def scalar(self, fmt, value, path):
        size = struct.calcsize(fmt)
        self.buf.extend(b'\x00' * (-(len(self.buf) - 4) % size))
        at = len(self.buf)
        self.buf.extend(struct.pack(self.endian + fmt, value))
        self.offsets.append((path, fmt, at))

    def write(self, value, kind='Contacts', path='root'):
        if isinstance(kind, tuple):
            self.scalar('I', len(value), path + '.count')
            for i, item in enumerate(value):
                self.write(item, kind[1], path + '[' + str(i) + ']')
        elif kind in SCHEMA:
            for field, typ in SCHEMA[kind]:
                self.write(getattr(value, field), typ, path + '.' + field)
        elif kind == 'string':
            encoded = value.encode('utf-8') + b'\x00'
            self.scalar('I', len(encoded), path + '.strlen')
            at = len(self.buf)
            self.buf.extend(encoded)
            self.offsets.append((path, 'string', at))
        else:
            self.scalar(kind, value, path)
        return bytes(self.buf)

def full_message(seed=0, names=None):
    rng = random.Random(seed)
    message = Contacts()
    message.header.stamp.sec = 10
    message.header.stamp.nanosec = seed % 1_000_000_000
    message.header.frame_id = ['world', '', 'µ_frame/棚'][seed % 3]
    names = names or ['x', 'yz', 'abc', '1234567', 'µ棚', '']
    values = [0., -0., 5e-324, -5e-324, 1e-200, 1e200,
              float('inf'), -float('inf'), struct.unpack('>d', bytes.fromhex('7ff8000000000042'))[0]]
    for i in range(1 + seed % 3):
        contact = Contact()
        for j, entity in enumerate((contact.collision1, contact.collision2)):
            entity.id = (2**64 - 1) if j else rng.getrandbits(64)
            entity.name = names[(seed + i + j) % len(names)]
            entity.type = 255 if j else 0
        for attr, count in [('positions', i + 1), ('normals', (i + seed) % 3)]:
            for k in range(count):
                point = Vector3()
                point.x, point.y, point.z = (values[(seed + k) % len(values)], rng.uniform(-10., 10.), -0.)
                getattr(contact, attr).append(point)
        contact.depths = array.array('d', [values[(seed + i + k) % len(values)] for k in range(i + 2)])
        for k in range(1 + (seed + i) % 3):
            wrench = JointWrench()
            wrench.header.stamp.sec = -1 if k == 0 else 2**31 - 1
            wrench.header.stamp.nanosec = 2**32 - 1
            wrench.header.frame_id = names[(i + k) % len(names)]
            wrench.body_1_name.data = names[(i + k + 1) % len(names)]
            wrench.body_2_name.data = names[(i + k + 2) % len(names)]
            wrench.body_1_id.data = 2**32 - 1
            wrench.body_2_id.data = rng.getrandbits(32)
            for side, body in enumerate((wrench.body_1_wrench, wrench.body_2_wrench)):
                for part, vec in enumerate((body.force, body.torque)):
                    vec.x, vec.y, vec.z = (values[(seed + i + k + side + part) % len(values)], rng.uniform(-10., 10.), -0.)
            contact.wrenches.append(wrench)
        message.contacts.append(contact)
    return message

def frozen_state(node):
    return fixture.freeze(dict(
        fields={k: getattr(node, k, '<absent>') for k in fixture.STATE_FIELDS},
        cancelled=node._cancel.is_set(), events=node.events, published=node.published,
        fine_deliveries=(len(node.fine_deliveries), node.fine_deliveries[-1:]),
        fine_holds=node.fine_holds))

class CallbackPair:
    def __init__(self, mode='stock', target=fixture.MODEL, bin_only=False):
        self.nodes = []
        self.clocks = []
        self.namespaces = []
        self.callbacks = []
        for side in range(2):
            node, clock = fixture.make_node(mode, target)
            node._fine_gripper_controller.observe_contacts = lambda msg, node=node: node.fine_deliveries.append(canonical(msg))
            if bin_only:
                fn = lambda msg, node=node: fixture.module._observe_place_contacts(node, msg, '/bin_contacts')
                ns = None
            else:
                method, ns = fixture.callback(fixture.CANDIDATE)
                fn = lambda msg, method=method, node=node: method(node, msg)
            self.nodes.append(node)
            self.clocks.append(clock)
            self.namespaces.append(ns)
            self.callbacks.append((lambda data, fn=fn: fn(deserialize_message(data, Contacts))) if side == 0
                                  else raw.raw_contacts_callback(node, fn, '/bin_contacts' if bin_only else '/contacts'))

    def equal(self):
        assert frozen_state(self.nodes[0]) == frozen_state(self.nodes[1])

    def deliver(self, message):
        data = serialize_message(message)
        assert canonical(deserialize_message(data, Contacts)) == canonical(raw.decode_contacts_cdr(data))
        for callback in self.callbacks:
            callback(data)
        self.equal()

    def reset(self, **kwargs):
        for node in self.nodes:
            node._clear_target_contact_samples(**kwargs)
        self.equal()

def stream_messages(count=96):
    f = fixture
    rng = random.Random(450052)
    messages = []
    for i in range(count):
        stamp = f.NOW + (i if i % 4 else rng.randrange(i + 1)) * 1_000_000
        entries = []
        for _ in range(rng.randrange(4)):
            finger = rng.choice((f.LEFT, f.INNER, f.RIGHT))
            first, second = (finger, f.BOOK) if rng.randrange(2) else (f.BOOK, finger)
            entries.append((first, second, rng.choice((0., 1., 3., 5., 8.1))))
        message = f.make_message(stamp, *entries)
        # Include valid unconsumed data so benchmark is not gripper-force-only.
        for c in message.contacts:
            point = Vector3(); point.x, point.y, point.z = .1, -.2, .3
            c.positions.append(point); c.normals.append(point)
            c.depths = array.array('d', [.001])
        messages.append(message)
    return messages
