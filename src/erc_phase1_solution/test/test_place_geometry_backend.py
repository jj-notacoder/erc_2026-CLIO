"""Node opt-in, managed result admission and owned-child reporting behavior."""
import threading
from types import SimpleNamespace

import pytest

from erc_phase1_solution import place_geometry_backend as wiring
from erc_phase1_solution import geometry_process_parent as parent
from erc_phase1_solution.geometry_process_pool import _process_stat_identity, GeometryProcessError


def node(enabled=True):
    result=SimpleNamespace(place_parallel_geometry_enabled=enabled,_lock=threading.Lock(),
        _place_parallel_geometry_identity=object(),_place_parallel_geometry_epoch=0,
        _place_parallel_geometry_active=False,events=[])
    result._publish_status=lambda event,**fields:result.events.append((event,fields))
    return result


@pytest.mark.parametrize('value',[None,0,1,0.,1.,'true','false',[],{}])
def test_explicit_nonboolean_selection_is_rejected(value):
    with pytest.raises(ValueError,match='Boolean'):
        wiring.plan_node_scene_checked_place(node(value),lambda *a,**k:pytest.fail('planner ran'))


@pytest.mark.parametrize('absent',[False,True])
def test_default_path_preserves_callable_arguments_and_result_identity(absent,monkeypatch):
    owner=node(False)
    if absent:del owner.place_parallel_geometry_enabled
    monkeypatch.setattr(parent,'ProcessPlanningBackend',lambda *a,**k:pytest.fail('pool constructed'))
    first,second,result=object(),object(),object();calls=[]
    def local(*args,**kwargs):calls.append((args,kwargs));return result
    assert wiring.plan_node_scene_checked_place(owner,local,first,target=second) is result
    assert calls==[((owner,first),dict(target=second))] and owner.events==[]


def test_disabled_initialization_resolves_no_packages():
    owner=node(False)
    wiring.initialize_place_geometry_backend(owner,lambda name:pytest.fail('package resolved'))
    assert owner._place_parallel_geometry_identity is None
    assert owner._place_parallel_geometry_epoch==0 and not owner._place_parallel_geometry_active


def test_enabled_initialization_uses_normal_resolved_shares_once(monkeypatch):
    import erc_phase1_solution.installed_geometry_identity as installed
    calls=[];identity=object();owner=node()
    monkeypatch.setattr(installed,'required_package_names',lambda description:('erc_description','arm'))
    class Identity:
        @staticmethod
        def capture(**kwargs):calls.append(kwargs);return identity
    monkeypatch.setattr(installed,'InstalledGeometryIdentity',Identity)
    wiring.initialize_place_geometry_backend(owner,lambda name:'/normal/share/'+name)
    assert calls==[dict(package_shares={'erc_description':'/normal/share/erc_description','arm':'/normal/share/arm'})]
    assert owner._place_parallel_geometry_identity is identity


def backend_fixture(monkeypatch,owner,*,exit_error=False,constructor_error=False):
    trace=[]
    identity=dict(pid=123,start_ticks=456,boot_id='boot',parent_pid=100,
                  process_group=123,session=123,argv=['python','-m','worker'])
    status=dict(workers_created=1,workers_reaped=1,children_closed=True,
        workers=[dict(pid=123,identity=identity,reaped=True,remaining_owned_child=False,returncode=-15)],
        pool_startup_wall_seconds=.1,sequence_wall_seconds=.2,query_statistics=dict(consumed=4))
    class Backend:
        def __init__(self,actual,actual_identity,*,epoch):
            assert actual is owner and actual_identity is owner._place_parallel_geometry_identity
            assert owner._place_parallel_geometry_active
            trace.append(('construct',epoch));self.pool=None;self.fallback_reason=None
            if constructor_error:
                error=RuntimeError('bootstrap failed');error.geometry_worker_status=status;raise error
        def __enter__(self):trace.append('enter');return self
        def __exit__(self,kind,error,tb):
            trace.append('exit');self.pool=SimpleNamespace(worker_status=lambda:status)
            if exit_error:raise RuntimeError('completion admission failed')
    monkeypatch.setattr(parent,'ProcessPlanningBackend',Backend)
    return trace,status


def test_selected_plan_returns_only_after_managed_exit_and_reports_owned_identity(monkeypatch):
    owner=node();trace,status=backend_fixture(monkeypatch,owner);result=object();first=object()
    def planner(actual,arg,*,geometry_backend,keyword):
        assert actual is owner and arg is first and keyword==2
        assert geometry_backend is not None
        trace.append('plan');return result
    assert wiring.plan_node_scene_checked_place(owner,planner,first,keyword=2) is result
    assert trace==[('construct',1),'enter','plan','exit']
    assert not owner._place_parallel_geometry_active
    event,fields=owner.events[0]
    assert event=='place_geometry_process_completed' and fields['passed']
    assert fields['workers']==status['workers'] and fields['children_closed']


@pytest.mark.parametrize('failure',['constructor','planner','exit'])
def test_failure_never_returns_plan_releases_active_and_preserves_closure(failure,monkeypatch):
    owner=node();trace,status=backend_fixture(monkeypatch,owner,
        constructor_error=failure=='constructor',exit_error=failure=='exit')
    def planner(*args,**kwargs):
        if failure=='planner':raise RuntimeError('planner failed')
        return object()
    with pytest.raises(RuntimeError):wiring.plan_node_scene_checked_place(owner,planner)
    assert not owner._place_parallel_geometry_active and owner._place_parallel_geometry_epoch==1
    fields=owner.events[-1][1]
    assert not fields['passed'] and fields['workers']==status['workers']
    assert fields['children_closed'] and fields['workers_reaped']==1


def test_failed_epoch_is_never_reused(monkeypatch):
    owner=node();epochs=[]
    class Backend:
        pool=None;fallback_reason=None
        def __init__(self,*args,epoch):epochs.append(epoch)
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(parent,'ProcessPlanningBackend',Backend)
    def broken(*a,**k):raise ValueError('failed')
    with pytest.raises(ValueError):wiring.plan_node_scene_checked_place(owner,broken)
    assert wiring.plan_node_scene_checked_place(owner,lambda *a,**k:7)==7
    assert epochs==[1,2]


@pytest.mark.parametrize('change',[{'_place_parallel_geometry_active':True},
    {'_place_parallel_geometry_epoch':True},{'_place_parallel_geometry_epoch':-1},
    {'_place_parallel_geometry_epoch':2**63-1},{'_place_parallel_geometry_identity':None}])
def test_invalid_or_concurrent_epoch_starts_no_backend(change,monkeypatch):
    owner=node();vars(owner).update(change)
    monkeypatch.setattr(parent,'ProcessPlanningBackend',lambda *a,**k:pytest.fail('constructed'))
    with pytest.raises(RuntimeError):wiring.plan_node_scene_checked_place(owner,lambda *a,**k:None)


def test_status_failure_does_not_hide_planning_failure(monkeypatch):
    owner=node();backend_fixture(monkeypatch,owner,exit_error=True)
    def failed_status(*a,**k):raise ValueError('telemetry failed')
    owner._publish_status=failed_status
    with pytest.raises(RuntimeError,match='completion admission'):
        wiring.plan_node_scene_checked_place(owner,lambda *a,**k:object())


@pytest.mark.parametrize('command',['python','name with spaces','worker ) ( unusual'])
def test_owned_proc_identity_parses_command_parentheses_without_field_shift(command):
    fields=['S','22','33','33']+['0']*15+['456']+['0']*10
    result=_process_stat_identity('33 ('+command+') '+' '.join(fields),33)
    assert result==dict(pid=33,parent_pid=22,process_group=33,session=33,start_ticks=456)


@pytest.mark.parametrize('raw',['99 (x) S 2 3','33 (x) S 2 3','33 (x) '+' '.join(['S']+['0']*19)])
def test_malformed_or_zero_start_identity_is_rejected(raw):
    with pytest.raises((GeometryProcessError,ValueError,IndexError)):_process_stat_identity(raw,33)


@pytest.mark.parametrize('kind',['valid','parent_changed','start_changed','wrong_argv','missing_boot'])
def test_worker_identity_admission_retains_exact_observation_on_mismatch(kind,monkeypatch):
    import erc_phase1_solution.geometry_process_pool as module
    parent_pid=module.os.getpid();calls=[0]
    def stat(start,parent):
        fields=['S',str(parent),'33','33']+['0']*15+[str(start)]+['0']*10
        return '33 (python3) '+' '.join(fields)
    expected=[module.sys.executable,'-u','-m','erc_phase1_solution.geometry_process_worker']
    class File:
        def __init__(self,name):self.name=str(name)
        def __truediv__(self,name):return File(self.name+'/'+str(name))
        def read_text(self):
            if self.name.endswith('/boot_id'):return '' if kind=='missing_boot' else 'boot\n'
            calls[0]+=1
            return stat(457 if kind=='start_changed' and calls[0]==2 else 456,
                        parent_pid+1 if kind=='parent_changed' else parent_pid)
        def read_bytes(self):
            return b'\0'.join(x.encode() for x in (['different'] if kind=='wrong_argv' else expected))+b'\0'
    monkeypatch.setattr(module,'Path',File)
    worker=module._Worker.__new__(module._Worker);worker.process=SimpleNamespace(pid=33);worker.identity=None
    if kind=='valid':
        worker.capture_identity()
        assert worker.identity['start_ticks']==456 and worker.identity['argv']==expected
    else:
        with pytest.raises(GeometryProcessError,match='identity changed'):worker.capture_identity()
        assert worker.identity is None
    assert worker.identity_observation['expected_argv']==expected
    assert worker.identity_observation['first']['pid']==33 and worker.identity_observation['last']['pid']==33
