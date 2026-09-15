"""Actual production main, stdlib doubles only; no ROS/model import or nodes."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import hashlib
import json
import unittest

PACKAGE=Path(__file__).parents[1]/'erc_phase1_solution/manipulation_node.py'
FIXTURE=Path(__file__).parent/'fixtures/manipulation_main_original.py'
PROVENANCE=FIXTURE.with_name('manipulation_main_original_provenance.json')


def main_ast(path):
    return next(x for x in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(x,ast.FunctionDef) and x.name=='main')


class ExternalShutdown(Exception): pass


def invoke(path, *, spin_error=None, okay=True, shutdown_error=None, args=None, marker_error=None):
    events=[]; state={'okay':okay}
    class Node:
        def __init__(self):
            events.append(('node',))
            self._cancel=SimpleNamespace(set=lambda:events.append(('cancel',)))
        def destroy_node(self): events.append(('destroy',))
    class Executor:
        def __init__(self,kind,*a,**kw):
            events.append(('executor',kind,a,kw))
        def add_node(self,node): events.append(('add',isinstance(node,Node)))
        def spin(self):
            events.append(('spin',))
            if spin_error is not None: raise spin_error
        def shutdown(self):
            events.append(('executor_shutdown',))
            if shutdown_error is not None: raise shutdown_error
    def shutdown():
        events.append(('rclpy_shutdown',));state['okay']=False
    def marker(node):
        if marker_error is not None: raise marker_error
    scope=dict(_mark_planning_cpu_executor=marker, rclpy=SimpleNamespace(init=lambda **kw:events.append(('init',kw)),
        ok=lambda:state['okay'],shutdown=shutdown),ManipulationNode=Node,
        MultiThreadedExecutor=lambda *a,**kw:Executor('multi4',*a,**kw),
        SingleThreadedExecutor=lambda *a,**kw:Executor('single',*a,**kw),
        ExternalShutdownException=ExternalShutdown)
    exec(compile(ast.Module(body=[main_ast(path)],type_ignores=[]),str(path),'exec'),scope)
    error=None;result=None
    try: result=scope['main'](args=args)
    except BaseException as exc: error=exc
    return events,result,error


class SingleExecutorMain(unittest.TestCase):
    def compare(self,**kwargs):
        old=invoke(FIXTURE,**kwargs);new=invoke(PACKAGE,**kwargs)
        before=[x for x in old[0] if x[0]!='executor']
        after=[x for x in new[0] if x[0]!='executor']
        self.assertEqual(before,after)
        self.assertIs(old[1],new[1]);self.assertIs(old[2],new[2])
        self.assertEqual([x for x in old[0] if x[0]=='executor'],[('executor','multi4',(),{'num_threads':4})])
        self.assertEqual([x for x in new[0] if x[0]=='executor'],[('executor','single',(),{})])
        return new

    def test_original_fixture_exact_ast_identity(self):
        pins=json.loads(PROVENANCE.read_text())
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),pins['fixture_sha256'])
        self.assertEqual(hashlib.sha256(ast.dump(main_ast(FIXTURE),include_attributes=False).encode()).hexdigest(),
                         pins['original_main_ast_sha256'])

    def test_only_executor_and_best_effort_marker_change_main(self):
        actual=copy.deepcopy(main_ast(PACKAGE))
        expected=ast.parse('try:\n    _mark_planning_cpu_executor(node)\nexcept Exception:\n    pass').body[0]
        markers=[x for x in actual.body if isinstance(x,ast.Try)
                 and ast.dump(x)==ast.dump(expected)]
        self.assertEqual(len(markers),1)
        actual.body.remove(markers[0])
        matches=[x for x in actual.body if isinstance(x,ast.Assign) and isinstance(x.value,ast.Call)
                 and isinstance(x.value.func,ast.Name) and x.value.func.id=='SingleThreadedExecutor']
        self.assertEqual(len(matches),1)
        self.assertEqual(matches[0].value.args,[]);self.assertEqual(matches[0].value.keywords,[])
        matches[0].value=ast.parse('MultiThreadedExecutor(num_threads=4)',mode='eval').body
        self.assertEqual(ast.dump(actual),ast.dump(main_ast(FIXTURE)))

    def test_production_import_is_true_single_not_one_thread_pool(self):
        imports=[x for x in ast.parse(PACKAGE.read_text()).body
                 if isinstance(x,ast.ImportFrom) and x.module=='rclpy.executors']
        self.assertEqual(len(imports),1)
        names={x.name for x in imports[0].names}
        self.assertIn('SingleThreadedExecutor',names);self.assertNotIn('MultiThreadedExecutor',names)

    def test_normal_spin_cancels_before_executor_and_node_teardown(self):
        events,result,error=self.compare(args=['--ros-args','-p','use_sim_time:=true'])
        self.assertIsNone(error);self.assertIsNone(result)
        self.assertEqual([x[0] for x in events],['init','node','executor','add','spin','cancel',
            'executor_shutdown','destroy','rclpy_shutdown'])

    def test_diagnostic_marker_failure_preserves_lifecycle(self):
        _,_,error=self.compare(marker_error=RuntimeError('optional marker failed'))
        self.assertIsNone(error)

    def test_none_arguments_are_forwarded_unchanged(self):
        events,_,_=self.compare(args=None)
        self.assertEqual(events[0],('init',{'args':None}))

    def test_keyboard_interrupt_is_swallowed_after_same_cleanup(self):
        _,_,error=self.compare(spin_error=KeyboardInterrupt())
        self.assertIsNone(error)

    def test_external_shutdown_is_swallowed_after_same_cleanup(self):
        _,_,error=self.compare(spin_error=ExternalShutdown())
        self.assertIsNone(error)

    def test_live_context_runtime_error_propagates_same_object(self):
        failure=RuntimeError('executor failure')
        _,_,error=self.compare(spin_error=failure,okay=True)
        self.assertIs(error,failure)

    def test_shutdown_context_runtime_error_does_not_repeat_shutdown(self):
        events,_,error=self.compare(spin_error=RuntimeError('context shutdown'),okay=False)
        self.assertIsNone(error);self.assertNotIn(('rclpy_shutdown',),events)

    def test_other_spin_errors_retain_original_exception_and_cleanup(self):
        failure=ValueError('same exception')
        _,_,error=self.compare(spin_error=failure)
        self.assertIs(error,failure)

    def test_original_executor_shutdown_failure_behavior_is_unchanged(self):
        failure=RuntimeError('shutdown failed')
        events,_,error=self.compare(shutdown_error=failure)
        self.assertIs(error,failure);self.assertNotIn(('destroy',),events)


if __name__=='__main__': unittest.main()
