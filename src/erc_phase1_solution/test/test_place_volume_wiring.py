"""Explicit node/parameter wiring; the existing pool still owns all workers."""
from pathlib import Path
import ast
from types import SimpleNamespace as NS

import pytest

from erc_phase1_solution import place_geometry_backend as wiring
from erc_phase1_solution import geometry_process_parent as parent
from test_place_geometry_backend import node


ROOT=Path(__file__).resolve().parents[1]
FLAG='place_carried_volume_parallel_enabled'


@pytest.mark.parametrize('value',[None,0,1,0.,1.,'true',[],{}])
def test_nonboolean_volume_flag_cannot_reserve_epoch_or_construct_pool(value,monkeypatch):
    current=node();setattr(current,FLAG,value)
    monkeypatch.setattr(parent,'ProcessPlanningBackend',lambda *a,**kw:pytest.fail('backend created'))
    with pytest.raises(ValueError,match='Boolean'):
        wiring.plan_node_scene_checked_place(current,lambda *a,**kw:pytest.fail('planner called'))
    assert current._place_parallel_geometry_epoch==0 and not current._place_parallel_geometry_active


@pytest.mark.parametrize('missing',[False,True])
def test_global_parallel_off_preserves_exact_local_call_even_with_volume_attribute(missing,monkeypatch):
    current=node(False)
    if not missing:setattr(current,FLAG,True)
    monkeypatch.setattr(parent,'ProcessPlanningBackend',lambda *a,**kw:pytest.fail('backend created'))
    sentinel=object();calls=[]
    def planner(*args,**kwargs):calls.append((args,kwargs));return sentinel
    assert wiring.plan_node_scene_checked_place(current,planner,'input',aperture=.017) is sentinel
    assert calls==[((current,'input'),{'aperture':.017})]
    assert current._place_parallel_geometry_epoch==0


@pytest.mark.parametrize('selected',[False,True])
@pytest.mark.parametrize('workers',[4,8])
def test_one_existing_managed_pool_receives_only_explicit_true_and_original_worker_count(monkeypatch,selected,workers):
    current=node();setattr(current,FLAG,selected);current.geometry_process_workers=workers
    events=[];result=object()
    class Managed:
        pool=None;fallback_reason=None
        def __init__(self,owner,identity,*,epoch,**kwargs):
            assert owner is current and identity is current._place_parallel_geometry_identity
            assert owner._place_parallel_geometry_active and epoch==1
            expected={} if workers==4 else {'worker_count':workers}
            if selected:expected['include_carried_volume']=True
            assert kwargs==expected;events.append(('construct',kwargs))
        def __enter__(self):events.append('enter');return self
        def __exit__(self,*args):events.append('reaped')
    monkeypatch.setattr(parent,'ProcessPlanningBackend',Managed)
    def planner(owner,value,*,geometry_backend):
        assert owner is current and type(geometry_backend) is Managed and value==42
        events.append('plan');return result
    assert wiring.plan_node_scene_checked_place(current,planner,42) is result
    assert events[1:]==['enter','plan','reaped']
    assert sum(isinstance(x,tuple) and x[0]=='construct' for x in events)==1
    assert current._place_parallel_geometry_epoch==1 and not current._place_parallel_geometry_active
    assert current.events[-1][1]['passed']


@pytest.mark.parametrize('failure',['planner','exit'])
def test_selected_volume_keeps_single_epoch_and_failure_cleanup(monkeypatch,failure):
    current=node();setattr(current,FLAG,True);events=[]
    class Managed:
        pool=None;fallback_reason=None
        def __init__(self,*args,**kwargs):assert kwargs==dict(epoch=1,include_carried_volume=True)
        def __enter__(self):events.append('enter');return self
        def __exit__(self,*args):
            events.append('closed')
            if failure=='exit':raise RuntimeError('exit failure')
    monkeypatch.setattr(parent,'ProcessPlanningBackend',Managed)
    def planner(*args,**kwargs):
        if failure=='planner':raise RuntimeError('planner failure')
        return object()
    with pytest.raises(RuntimeError,match='failure'):wiring.plan_node_scene_checked_place(current,planner)
    assert events==['enter','closed']
    assert current._place_parallel_geometry_epoch==1 and not current._place_parallel_geometry_active
    assert current.events[-1][1]['passed'] is False


def constructor_fragments():
    source=(ROOT/'erc_phase1_solution/manipulation_node.py').read_text()
    tree=ast.parse(source)
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ManipulationNode')
    init=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='__init__')
    assignment=next(x for x in init.body if isinstance(x,ast.Assign) and any(
        isinstance(t,ast.Attribute) and t.attr==FLAG for t in x.targets))
    index=init.body.index(assignment);guard=init.body[index+1]
    assert isinstance(guard,ast.If) and FLAG in ast.unparse(guard.test)
    declarations=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_declare_parameters')
    dictionaries=[x for x in ast.walk(declarations) if isinstance(x,ast.Dict)]
    defaults=[v for d in dictionaries for k,v in zip(d.keys,d.values) if isinstance(k,ast.Constant) and k.value==FLAG]
    assert len(defaults)==1 and ast.literal_eval(defaults[0]) is False
    return [assignment,guard]


@pytest.mark.parametrize('place,volume',[(False,False),(True,False),(True,True)])
def test_actual_constructor_assignment_default_and_existing_pool_dependency(place,volume):
    current=NS(place_parallel_geometry_enabled=place,get_parameter=lambda name:NS(value=volume))
    scope=dict(self=current,checked_place_carried_volume_parallel_enabled=wiring.checked_place_carried_volume_parallel_enabled)
    exec(compile(ast.fix_missing_locations(ast.Module(body=constructor_fragments(),type_ignores=[])),
        'actual_volume_constructor_assignment','exec'),scope)
    assert getattr(current,FLAG) is volume


def test_actual_constructor_refuses_volume_selection_without_existing_place_pool():
    current=NS(place_parallel_geometry_enabled=False,get_parameter=lambda name:NS(value=True))
    scope=dict(self=current,checked_place_carried_volume_parallel_enabled=wiring.checked_place_carried_volume_parallel_enabled)
    with pytest.raises(ValueError,match='requires parallel PLACE'):
        exec(compile(ast.fix_missing_locations(ast.Module(body=constructor_fragments(),type_ignores=[])),
            'actual_volume_constructor_dependency','exec'),scope)


def test_true_volume_flag_is_consumed_by_backend_not_forwarded_as_resource_option():
    backend=parent.ProcessPlanningBackend(object(),object(),epoch=1,include_carried_volume=True,worker_count=8)
    assert backend.include_carried_volume and backend.limits=={'worker_count':8}
    assert backend.pool is None and not backend.volume_active


def test_initialization_does_not_create_additional_identity_or_workers_for_volume_flag(monkeypatch):
    import erc_phase1_solution.installed_geometry_identity as installed
    current=node();setattr(current,FLAG,True);captures=[];identity=object()
    monkeypatch.setattr(installed,'required_package_names',lambda description:('erc_description',))
    class Identity:
        @staticmethod
        def capture(**kwargs):captures.append(kwargs);return identity
    monkeypatch.setattr(installed,'InstalledGeometryIdentity',Identity)
    monkeypatch.setattr(parent,'ProcessPlanningBackend',lambda *a,**kw:pytest.fail('constructor made pool'))
    wiring.initialize_place_geometry_backend(current,lambda name:'/share/'+name)
    assert captures==[dict(package_shares={'erc_description':'/share/erc_description'})]
    assert current._place_parallel_geometry_identity is identity
    assert current._place_parallel_geometry_epoch==0 and not current._place_parallel_geometry_active


def test_custom_volume_predicate_keeps_original_scene_pool_and_serial_volume_fallback(monkeypatch):
    from erc_phase1_solution import exact_world_geometry
    current=node();current._cancel=NS(is_set=lambda:False);state=NS(node=current);created=[]
    # Isolate the volume-specific method gate after the existing ordinary
    # scene/model gate. A custom payload predicate must not gain the fast path.
    monkeypatch.setattr(parent,'ordinary_scene_components',lambda value:value is state)
    monkeypatch.setattr(exact_world_geometry,'ordinary_model_producer',lambda value:value is current)
    monkeypatch.setattr(parent,'scene_model_signature',lambda value:'d'*64)
    class Capture:
        reference_copy={}
        def __init__(self,owner,actual,description):
            assert owner is current and actual is state
            assert 'carried_volume_context' not in description
        def verify_bindings(self):pass
    class Pool:
        def __init__(self,identity,**kwargs):created.append(kwargs)
        def finish(self):pass
        def close(self):pass
    monkeypatch.setattr(parent,'PairedSceneCapture',Capture)
    monkeypatch.setattr(parent,'GeometryProcessPool',Pool)
    monkeypatch.setattr(parent,'measured_scene_context',lambda *args:{})
    identity=NS(verify=lambda:None)
    with parent.ProcessPlanningBackend(current,identity,epoch=1,include_carried_volume=True) as backend:
        result=backend.attach(state,dict(bin_scene={},table_scene={}),volume_aperture=.017)
        assert type(result) is parent.ProcessPlaceChecker
        assert not backend.volume_active and backend.volume_fallback_reason=='unsupported custom volume predicates'
        assert not hasattr(state,'_place_volume_context')
    assert len(created)==1 and 'carried_volume_context' not in created[0]['scene']
