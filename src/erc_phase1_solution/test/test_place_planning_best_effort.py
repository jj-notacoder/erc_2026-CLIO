"""Telemetry may not replace admission errors; no meshes, nodes or commands."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from erc_phase1_solution import scene_checked_place as geometry


class DiagnosticReadError(RuntimeError):
    pass


class RaisingFields(SimpleNamespace):
    def __getattribute__(self, name):
        if name in object.__getattribute__(self, '__dict__').get('_diagnostic_raise', ()):
            raise DiagnosticReadError(name)
        return super().__getattribute__(name)


def node_for_admission():
    feedback=RaisingFields(position=.017, stamp_ns=123)
    node=RaisingFields(_cancel=threading.Event(), bin_scene_required=True,
        gripper_open=.069, _held_book_corners=np.zeros((8,3)),
        _publish_status=Mock(), _adaptive_motion_feedback=lambda:(feedback,None))
    return node,feedback


def require_bin_error(node,monkeypatch):
    loader=Mock(side_effect=AssertionError('admission must precede mesh load'))
    monkeypatch.setattr(geometry,'load_stl_triangles',loader)
    with pytest.raises(RuntimeError,match='^scene_checked_place_bin_pose_required$'):
        geometry.plan_scene_checked_place(node,[],np.eye(3),None,None,None,[.9,0,.75],
                                         lambda _: '/unused',table_scene={})
    loader.assert_not_called()


@pytest.mark.parametrize('missing',['held','stamp','both'])
def test_optional_initial_field_absence_preserves_required_bin_error(monkeypatch,missing):
    node,feedback=node_for_admission()
    if missing in ('held','both'): del node._held_book_corners
    if missing in ('stamp','both'): del feedback.stamp_ns
    require_bin_error(node,monkeypatch)


@pytest.mark.parametrize('field',['_held_book_corners','_selected_place_scene_reference','_publish_status','stamp_ns'])
def test_raising_diagnostic_attribute_preserves_required_bin_error(monkeypatch,field):
    node,feedback=node_for_admission()
    (feedback if field=='stamp_ns' else node)._diagnostic_raise={field}
    require_bin_error(node,monkeypatch)


def test_publisher_lookup_is_best_effort_for_all_stages():
    tree=ast.parse(Path(geometry.__file__).read_text())
    plan=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='plan_scene_checked_place')
    progress=next(n for n in plan.body if isinstance(n,ast.FunctionDef) and n.name=='progress')
    node=RaisingFields(_diagnostic_raise={'_publish_status'})
    namespace=dict(node=node,table_scene={},deepcopy=deepcopy,np=np)
    exec(compile(ast.Module(body=[progress],type_ignores=[]),'actual_progress','exec'),namespace)
    namespace['progress']('candidate_accepted',scene_samples=1)


def test_optional_search_budget_read_is_best_effort():
    tree=ast.parse(Path(geometry.__file__).read_text())
    plan=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='plan_scene_checked_place')
    block=next(n for n in plan.body if isinstance(n,ast.Try) and any(
        isinstance(v,ast.Constant) and v.value=='cartesian_search_begin' for v in ast.walk(n)))
    progress=Mock()
    namespace=dict(progress=progress,node=RaisingFields(_diagnostic_raise={'scene_cartesian_wall_seconds'}),scene=SimpleNamespace(samples=61))
    exec(compile(ast.Module(body=[block],type_ignores=[]),'actual_budget_stage','exec'),namespace)
    progress.assert_not_called()


def test_saved_valid_run37_initial_payload_is_identical_before_geometry(monkeypatch):
    fixture=json.loads((Path(__file__).parent/'fixtures/run37_planning_initial_fields.json').read_text())
    original=fixture['expected_fields']
    expected=deepcopy(original); expected.pop('planning_wall_seconds')
    feedback=SimpleNamespace(position=original['measured_master'],stamp_ns=original['master_producer_stamp_ns'])
    records=[]
    node=SimpleNamespace(_cancel=threading.Event(),gripper_open=.069,bin_scene_required=True,
        _adaptive_motion_feedback=lambda:(feedback,None),
        _held_book_corners=np.asarray(original['held_book_corners']),
        _selected_place_scene_reference=deepcopy(original['selected_admission_reference']),
        _publish_status=lambda event,**fields:records.append((event,fields)))
    sentinel=RuntimeError('stop at original resolver boundary before mesh access')
    def stop(_): raise sentinel
    loader=Mock(side_effect=AssertionError('unexpected mesh load'))
    monkeypatch.setattr(geometry,'load_stl_triangles',loader)
    with pytest.raises(RuntimeError) as caught:
        geometry.plan_scene_checked_place(node,np.asarray(original['cartesian_positions']),
            np.asarray(original['tool_rotation']),np.asarray(original['actual_carry_start']),
            np.asarray(original['torso_ready']),np.asarray(original['staging_seed']),
            original['bin_floor_point'],stop,table_scene=original['selected_table_scene'],
            bin_scene=original['selected_bin_scene'])
    assert caught.value is sentinel
    loader.assert_not_called()
    assert len(records)==1 and records[0][0]=='place_planning_stage'
    event=records[0][1]; assert event.pop('planning_wall_seconds')>=0
    assert event==expected
    json.dumps(event,allow_nan=False)
    event['selected_table_scene']['origin'][0]=999
    assert original['selected_table_scene']['origin'][0]!=999


@pytest.mark.parametrize('fault',['conversion','publisher'])
def test_initial_logging_failure_cannot_replace_original_resolver_error(fault):
    node,_=node_for_admission()
    sentinel=RuntimeError('original resolver failure')
    if fault=='conversion':
        class BadArray:
            def __array__(self,*args,**kwargs): raise DiagnosticReadError('conversion')
        node._held_book_corners=BadArray()
    else:
        node._publish_status=Mock(side_effect=DiagnosticReadError('publisher'))
    def stop(_): raise sentinel
    with pytest.raises(RuntimeError) as caught:
        geometry.plan_scene_checked_place(node,[],np.eye(3),None,None,None,[.9,0,.75],stop,table_scene={})
    assert caught.value is sentinel
