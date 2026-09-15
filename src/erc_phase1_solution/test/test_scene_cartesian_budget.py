"""Original-deadline behavior for the explicit development wall allowance."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_scene_cartesian_solver as base

m=base.m


class PlanningBudgetTests(unittest.TestCase):
    setUp=base.SolverTests.setUp
    run_solver=base.SolverTests.run_solver
    validator=base.SolverTests.validator

    def test_default_and_search_count_budgets_remain_unchanged(self):
        default=m.SearchLimits()
        self.assertEqual((default.wall_seconds,default.max_ik_calls,default.max_paths,default.max_candidates),
                         (420.,512,24,6))
        for seconds in (900.,1800.):
            selected=m.SearchLimits(wall_seconds=seconds)
            self.assertEqual((selected.max_ik_calls,selected.max_paths,selected.max_candidates),(512,24,6))

    def test_finite_upper_bound_accepts1800_and_rejects_invalid_before_ik(self):
        for seconds in (900.,1800.):
            with self.subTest(accepted=seconds):
                self.run_solver(limits=m.SearchLimits(wall_seconds=seconds))
        for seconds in (math.nextafter(1800.,math.inf),1800.000001,0.,-1.,float('nan'),float('inf'),-float('inf')):
            self.chain.calls.clear()
            with self.subTest(rejected=seconds),self.assertRaises(ValueError):
                self.run_solver(limits=m.SearchLimits(wall_seconds=seconds))
            self.assertEqual(self.chain.calls,[])

    def test_only_explicit_override_can_accept_candidate_after_default_or_prior_budget(self):
        # Run38's full candidate callback returned at 1316.044 search-wall seconds.
        for elapsed,limit,success in ((500.,420.,False),(500.,900.,True),
                                      (1316.044,900.,False),(1316.044,1800.,True)):
            self.setUp()
            clock=[0.]
            def validate(*args):clock[0]=elapsed;return True
            self.validator=validate
            info={}
            with self.subTest(elapsed=elapsed,limit=limit),patch.object(m.time,'monotonic',side_effect=lambda:clock[0]):
                if success:
                    self.run_solver(limits=m.SearchLimits(wall_seconds=limit),diagnostics_out=info)
                    self.assertEqual(info['wall_budget_seconds'],limit)
                    self.assertEqual(info['wall_seconds'],elapsed)
                else:
                    with self.assertRaisesRegex(m.SceneCartesianSearchError,'wall_budget'):
                        self.run_solver(limits=m.SearchLimits(wall_seconds=limit))

    def test_exact_expiry_after_successful_full_validator_discards_plan(self):
        for limit in (900.,1800.):
            self.setUp()
            clock=[0.]
            def validate(*args):clock[0]=limit;return True
            self.validator=validate
            with self.subTest(limit=limit),patch.object(m.time,'monotonic',side_effect=lambda:clock[0]):
                with self.assertRaisesRegex(m.SceneCartesianSearchError,'wall_budget'):
                    self.run_solver(limits=m.SearchLimits(wall_seconds=limit))

    def test_original_deadline_survives_rejected_candidate_and_next_branch(self):
        for limit,first,second in ((900.,500.,400.),(1800.,1000.,800.)):
            self.setUp()
            clock=[0.];seen=[]
            self.positions=self.positions[:1]
            def validate(path,transition):
                seen.append(path[-1].copy())
                clock[0]+=first if len(seen)==1 else second
                return len(seen)==2
            self.validator=validate
            with self.subTest(limit=limit),patch.object(m.time,'monotonic',side_effect=lambda:clock[0]):
                with self.assertRaisesRegex(m.SceneCartesianSearchError,'wall_budget'):
                    self.run_solver(limits=m.SearchLimits(wall_seconds=limit))
            self.assertEqual(len(seen),2)

    def test_original_deadline_includes_all_ik_callbacks(self):
        for limit in (900.,1800.):
            self.setUp()
            clock=[0.]
            def modify(q,*args):clock[0]+=limit/3.;return q
            self.chain.modifier=modify
            with self.subTest(limit=limit),patch.object(m.time,'monotonic',side_effect=lambda:clock[0]):
                with self.assertRaisesRegex(m.SceneCartesianSearchError,'wall_budget'):
                    self.run_solver(limits=m.SearchLimits(wall_seconds=limit))
            self.assertEqual(len(self.chain.calls),3)

    def test_cancellation_during_extended_full_validation_still_rejects(self):
        for limit in (900.,1800.):
            self.setUp()
            clock=[0.]
            def validate(*args):clock[0]=limit*.75;self.node._cancel.set();return True
            self.validator=validate
            with self.subTest(limit=limit),patch.object(m.time,'monotonic',side_effect=lambda:clock[0]):
                with self.assertRaisesRegex(m.SceneCartesianSearchError,'cancelled'):
                    self.run_solver(limits=m.SearchLimits(wall_seconds=limit))

    def test_actual_node_parameter_default_and_validation_statements(self):
        source=Path(__file__).resolve().parents[1]/'erc_phase1_solution/manipulation_node.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ManipulationNode')
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        index=next(i for i,n in enumerate(init.body) if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Attribute) and t.attr=='scene_cartesian_wall_seconds' for t in n.targets))
        assert isinstance(init.body[index+1],ast.If)
        code=compile(ast.Module(body=init.body[index:index+2],type_ignores=[]),'node_budget_init','exec')
        defaults=next(n for n in ast.walk(cls) if isinstance(n,ast.Dict)
                      and any(isinstance(k,ast.Constant) and k.value=='scene_cartesian_wall_seconds' for k in n.keys))
        default=next(ast.literal_eval(v) for k,v in zip(defaults.keys,defaults.values)
                     if isinstance(k,ast.Constant) and k.value=='scene_cartesian_wall_seconds')
        self.assertEqual(default,420.)
        for value in (default,900.,1800.):
            node=SimpleNamespace(get_parameter=lambda name:SimpleNamespace(value=value))
            exec(code,dict(self=node,math=math))
            self.assertEqual(node.scene_cartesian_wall_seconds,value)
        for value in (0.,-1.,math.nextafter(1800.,math.inf),1800.000001,float('nan'),float('inf'),-float('inf')):
            node=SimpleNamespace(get_parameter=lambda name:SimpleNamespace(value=value))
            with self.subTest(value=value),self.assertRaises(ValueError):exec(code,dict(self=node,math=math))


if __name__=='__main__':unittest.main(verbosity=2)
