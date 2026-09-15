"""Actual raw subscription/command AST and permanent malformed-wire interlocks.

No ROS executor or geometry is loaded. Serialization/callback-state equivalence
has a separate real ROS oracle; these cases isolate command and lock behavior.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import threading
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT/'erc_phase1_solution/manipulation_node.py'
ORIGINAL = ROOT/'test/fixtures/raw_contacts_original_command.py'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


raw = load_module('_raw_contacts_integration_codec', ROOT/'erc_phase1_solution/raw_contacts.py')
support = load_module('_raw_contacts_integration_source', ROOT/'test/candidate_composition_support.py')
TREE = ast.parse(NODE.read_text())
OWNER = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'ManipulationNode')
METHODS = {n.name:n for n in OWNER.body if isinstance(n, ast.FunctionDef)}


def empty_message():
    # CDR1 LE, Header(sec=1,nanosec=0,frame_id=''), then zero contacts.
    return b'\x00\x01\x00\x00' + struct.pack('<iiI', 1, 0, 1) + b'\0'*4 + struct.pack('<I', 0)


def fake_node():
    node = NS(_lock=threading.Lock(), _adaptive_command_lock=threading.RLock(),
              _cancel=threading.Event(), _busy=False, _goal_handles=[],
              _pending_retained_acceptances=[], events=[], holds=[], starts=[],
              _held_book_corners=None, _transport_lock_engaged=False,
              _adaptive_close_active=False, _adaptive_hold_sent=False,
              delivery_evidence_enabled=False)
    node._adaptive_command_guard = lambda: node._adaptive_command_lock
    node._hold_adaptive_gripper = lambda reason: node.holds.append(reason)
    node._publish_status = lambda event, **kw: node.events.append((event, kw))
    node._run_command = lambda *args: None
    return node


def command_method(node, original=False):
    method = (next(n for n in ast.parse(ORIGINAL.read_text()).body if isinstance(n, ast.FunctionDef))
              if original else METHODS['_on_command'])
    class Thread:
        def __init__(self, **kwargs):
            self.kwargs=kwargs
        def start(self):
            node.starts.append(self.kwargs['args'])
    scope = dict(__package__="erc_phase1_solution",decode_event=json.loads, _stock_diagnostic=lambda n:False,
                 threading=NS(Thread=Thread),
                 _planning_cpu_command_target=lambda n, fn:fn)
    module=ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(NODE), 'exec'), scope)
    return lambda name:scope['_on_command'](node, NS(data=json.dumps({'event':name})))


class RawContactsIntegrationTests(unittest.TestCase):
    def test_exact_inverse_retains_entire_r52_node(self):
        restored=support.restore_raw_contacts_source(NODE.read_text())
        self.assertEqual(hashlib.sha256(restored.encode()).hexdigest(),
                         '06bef5efb58b73301aced8e942a1df5528a931a3d2ce46e74d307fa29a5810bf')
        self.assertEqual(hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                         '495c89043341871cdd41c525795e13964e8f830e364947fb6eef9d761e61cc69')

    def test_inverse_rejects_unreviewed_mutation_or_missing_guard(self):
        for changed in (NODE.read_text().replace("reason='raw_contacts_decode_failed'", "reason='ignored'"),
                        NODE.read_text().replace('self._cancel.clear()', 'self._cancel.set()')):
            with self.assertRaises(AssertionError):
                support.restore_raw_contacts_source(changed)

    def test_default_false_and_actual_parameter_property(self):
        declared=[n for n in ast.walk(METHODS['_declare_parameters']) if isinstance(n,ast.Dict)
                  and any(isinstance(k,ast.Constant) and k.value=='raw_contacts_enabled' for k in n.keys)]
        self.assertEqual(len(declared),1)
        values={k.value:v for k,v in zip(declared[0].keys,declared[0].values) if isinstance(k,ast.Constant)}
        self.assertIs(values['raw_contacts_enabled'].value,False)
        assignment=next(n for n in METHODS['__init__'].body if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Attribute) and t.attr=='raw_contacts_enabled' for t in n.targets))
        for value in (False,True):
            node=NS(get_parameter=lambda key:NS(value=value) if key=='raw_contacts_enabled' else self.fail(key))
            exec(compile(ast.Module(body=[assignment],type_ignores=[]),str(NODE),'exec'),{'self':node})
            self.assertIs(node.raw_contacts_enabled,value)

    def test_actual_subscriptions_same_topic_order_qos_and_routing(self):
        init=METHODS['__init__']
        contacts=next(n for n in init.body if isinstance(n,ast.If)
                      and ast.unparse(n.test)=="getattr(self, 'raw_contacts_enabled', False)")
        bins=next(n for n in init.body if isinstance(n,ast.If)
                  and ast.unparse(n.test)=='self.table_scene_required' and "'/bin_contacts'" in ast.unparse(n))
        for enabled in (None,False,True):
            for table in (False,True):
                with self.subTest(raw=enabled,table=table):
                    node=fake_node();node.table_scene_required=table
                    if enabled is not None:node.raw_contacts_enabled=enabled
                    subscriptions=[];calls=[];roles=[];qos=object();contact_type=object()
                    node.create_subscription=lambda *a,**kw:subscriptions.append((a,kw))
                    node._on_contacts=lambda msg:calls.append(('/contacts',msg)) or 'grip_return'
                    def diagnostic(n,role,fn):roles.append(role);return fn
                    scope=dict(self=node,Contacts=contact_type,SENSOR_QOS=qos,
                               _planning_cpu_callback=diagnostic,_raw_contacts_callback=raw.raw_contacts_callback,
                               _observe_place_contacts=lambda n,msg,topic:calls.append((topic,msg)) or 'bin_return')
                    exec(compile(ast.Module(body=[contacts,bins],type_ignores=[]),str(NODE),'exec'),scope)
                    topics=['/contacts']+(['/bin_contacts'] if table else [])
                    self.assertEqual([a[1] for a,kw in subscriptions],topics)
                    self.assertEqual(roles,['_on_contacts']+(['bin_contacts'] if table else []))
                    for (args,kw),topic in zip(subscriptions,topics):
                        self.assertIs(args[0],contact_type);self.assertIs(args[3],qos)
                        self.assertEqual(kw,{'raw':True} if enabled else {})
                        msg=empty_message() if enabled else object()
                        value=args[2](msg)
                        self.assertEqual(value,'grip_return' if topic=='/contacts' else 'bin_return')
                        self.assertEqual(calls[-1][0],topic)
                        if enabled:self.assertEqual(calls[-1][1].header.stamp.sec,1)
                        else:self.assertIs(calls[-1][1],msg)
                    self.assertEqual(len(calls),len(topics))

    def test_no_fault_default_command_flow_matches_original(self):
        for busy,goals,pending in ((False,False,False),(True,False,False),(False,True,False),(False,False,True)):
            records=[]
            for original in (True,False):
                node=fake_node();node._busy=busy;node._cancel.set()
                node._goal_handles=[object()] if goals else []
                node._pending_retained_acceptances=[object()] if pending else []
                command_method(node,original)('place')
                records.append((node.events,node.starts,node._cancel.is_set(),node._busy))
            self.assertEqual(*records)

    def test_decode_fault_rejects_new_command_after_ordinary_latches_reset(self):
        node=fake_node();raw.raw_contacts_callback(node,lambda _:self.fail('partial callback'))(b'bad')
        first=node._raw_contacts_first_failure
        node._held_grip_sensor_fault=None;node._payload_hazard_latched=None;node._adaptive_overload_latched=None
        node._cancel.clear()  # Deliberately simulate another old reset path.
        command_method(node)('pick')
        self.assertFalse(node.starts);self.assertFalse(node._busy)
        self.assertIs(node._raw_contacts_first_failure,first)
        self.assertEqual(node.events[-1],('rejected',dict(command='pick',reason='raw_contacts_decode_failed')))

    def test_cancel_abort_stop_still_use_original_cancellation_branch(self):
        for command in ('cancel','abort','stop'):
            node=fake_node();node._raw_contacts_first_failure={'reason':'wire'}
            cancellations=[];node._goal_handles=[NS(cancel_goal_async=lambda:cancellations.append(True))]
            command_method(node)(command)
            self.assertTrue(node._cancel.is_set());self.assertEqual(cancellations,[True])
            self.assertEqual(node.events[-1],('cancelled',{}));self.assertFalse(node.starts)

    def test_second_commit_closes_command_clear_race(self):
        node=fake_node();entered=threading.Event();errors=[]
        class Event:
            def __init__(self):self.inner=threading.Event();self.count=0
            def set(self):self.count+=1;self.inner.set();entered.set()
            def clear(self):self.inner.clear()
            def is_set(self):return self.inner.is_set()
        node._cancel=Event()
        with node._lock:
            def receive():
                try:raw.raw_contacts_callback(node,lambda _:self.fail('partial'))(b'bad')
                except BaseException as error:errors.append(error)
            worker=threading.Thread(target=receive)
            worker.start();self.assertTrue(entered.wait(1.))
            node._cancel.clear()  # Equivalent to an admission already holding sensor lock.
        worker.join(1.)
        self.assertFalse(worker.is_alive());self.assertFalse(errors)
        self.assertEqual(node._cancel.count,2);self.assertTrue(node._cancel.is_set())
        self.assertIsNotNone(node._raw_contacts_first_failure)

    def test_permanent_fault_hold_and_evidence_are_bounded(self):
        node=fake_node();node._held_book_corners=object();node._adaptive_close_active=True
        adapter=raw.raw_contacts_callback(node,lambda _:None,'/bin_contacts')
        for _ in range(5):adapter(b'bad')
        self.assertEqual(node.holds,['raw_contacts_decode_failed'])
        self.assertEqual(len(node.events),1)
        self.assertEqual(node._raw_contacts_first_failure['source_topic'],'/bin_contacts')
        self.assertEqual(node._payload_hazard_latched,'raw_contacts_decode_failed')
        self.assertEqual(node._adaptive_overload_latched,'raw_contacts_decode_failed')

    def test_rejected_command_status_error_cannot_start_or_clear_cancel(self):
        node=fake_node();node._raw_contacts_first_failure={'reason':'wire'};node._cancel.set()
        error=RuntimeError('publisher')
        def publish(*a,**kw):raise error
        node._publish_status=publish
        with self.assertRaises(RuntimeError) as caught:command_method(node)('place')
        self.assertIs(caught.exception,error);self.assertFalse(node.starts);self.assertTrue(node._cancel.is_set())

    def test_valid_later_deliveries_preserve_callback_and_do_not_clear_wire_fault(self):
        node=fake_node();calls=[]
        adapter=raw.raw_contacts_callback(node,lambda msg:calls.append(msg) or 'result')
        adapter(b'bad');first=node._raw_contacts_first_failure
        self.assertEqual(adapter(empty_message()),'result')
        self.assertEqual(len(calls),1);self.assertIs(node._raw_contacts_first_failure,first)
        self.assertTrue(node._cancel.is_set())


if __name__=='__main__':
    unittest.main()
