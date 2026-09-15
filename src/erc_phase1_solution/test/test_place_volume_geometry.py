"""Exact carried-volume grids, pure operations and existing scene admissions."""
from dataclasses import replace
import copy
from types import SimpleNamespace as NS

import numpy as np
import pytest

from erc_phase1_solution import place_volume_geometry as volume
from erc_phase1_solution.geometry_process_parent import PairedSceneCapture, ProcessPlanningBackend
from erc_phase1_solution.geometry_process_pool import GeometryProcessError
from erc_phase1_solution.geometry_process_protocol import query_from_wire, query_to_wire, SampleDelta
from erc_phase1_solution.pure_geometry_owner import GeometryQuery, RobotGeometryOwner, RobotCollisionGeometry
from test_geometry_process_parent import sensor_fixture
from test_pickup_geometry_process import corners, Chain
from test_scene_checked_place_planner import planning


def query(operation='place_payload',index=0):
    return volume.PlaceVolumeQuery(GeometryQuery.capture(request_id=index,epoch=1,
        source_id='a'*64,model_id='b'*64,scene_id='c'*64,q=np.zeros(8),right=np.zeros(7),
        head=np.zeros(2),aperture=.017,loaded=True),operation)


@pytest.mark.parametrize('operation',['place_payload','place_body'])
def test_distinct_exact_operation_roundtrip(operation):
    original=query(operation)
    assert query_from_wire(query_to_wire(original))==original
    assert original.input_sha256!=original.sample.input_sha256
    other=replace(original,operation='place_body' if operation=='place_payload' else 'place_payload')
    assert original.input_sha256!=other.input_sha256
    message=query_to_wire(original);message['place_volume_operation']=other.operation
    with pytest.raises(ValueError,match='hash mismatch'):query_from_wire(message)


@pytest.mark.parametrize('operation',[None,False,[],{},'pickup_payload','place_scene'])
def test_unknown_operation_rejected(operation):
    with pytest.raises(ValueError):query(operation).validated()


def test_nested_and_unloaded_queries_cannot_enter_volume():
    original=query()
    with pytest.raises(ValueError,match='nested'):replace(original,sample=original).validated()
    with pytest.raises(ValueError,match='carried book'):
        replace(original,sample=replace(original.sample,loaded=False)).validated()
    message=query_to_wire(original);message['sample']=query_to_wire(original)
    with pytest.raises(ValueError,match='nested'):query_from_wire(message)


def context():
    return dict(aperture=.017,maximum_tilt=.6,supported_jaw_vertical_component=.7,
                joint_step=.25,orientation_step=.3)


@pytest.mark.parametrize('change',[dict(extra=0),dict(aperture=.07),dict(aperture=True),
    dict(maximum_tilt=float('nan')),dict(maximum_tilt=4.),dict(joint_step=0),
    dict(orientation_step=-.1),dict(supported_jaw_vertical_component=0)])
def test_context_rejects_unbound_or_invalid_fields(change):
    value=context();value.update(change)
    with pytest.raises(ValueError):volume.validate_volume_context(value)


class ImmediatePool:
    """Consume actual pure worker deltas; never substitutes collision verdicts."""
    def __init__(self,binding,owner):
        self.__dict__.update(binding.__dict__)
        self.owner=owner;self.queries=[];self.before_capture=lambda:None
    def evaluate_sequence(self,items,*,capture,consume):
        for item in items:
            self.before_capture()
            current=capture(len(self.queries),item);self.queries.append(current)
            if not consume(current,volume.evaluate_owned_volume(self.owner,current)):
                return False
        return True


def fixture(monkeypatch,count=61,reject_payload=None,reject_body=None):
    node,state,_,binding=sensor_fixture(count)
    node.chain=Chain()
    for key,name in volume.CONTEXT_FIELDS.items():setattr(node,name,context()[key])
    node._held_book_corners=corners();state.attached=node._held_book_corners
    state._place_volume_context=context()
    original_predicates=[]
    def payload(q,*args,**kw):
        original_predicates.append(('place_payload',q.copy(),copy.deepcopy(kw)))
        return 'payload_robot' if reject_payload is not None and q[1]>=reject_payload else None
    def body(q,**kw):
        original_predicates.append(('place_body',q.copy(),copy.deepcopy(kw)))
        return 'arm_head' if reject_body is not None and q[1]>=reject_body else None
    node._carried_robot_collision=payload;node._robot_self_collision=body
    node._resolved_right_positions=lambda q:np.asarray(q)
    node._resolved_head_positions=lambda q:np.asarray(q)
    owned=NS(_scene=state,_epoch=binding.epoch,_scene_id=binding.scene_id,
        assets=binding.identity,_last_request_id=-1,_queries_seen=0,_apertures=set(),
        maximum_queries=20000,maximum_distinct_apertures=128,chain=node.chain,_cancel=node._cancel,
        _carried_robot_collision=payload,_robot_self_collision=body)
    monkeypatch.setattr(volume,'ordinary_volume_parent',lambda current:current is node)
    description=dict(bin_scene=node._selected_place_bin_scene,table_scene=node._selected_place_table_scene,
        attached_corners=corners().tolist(),carried_volume_context=context())
    capture=PairedSceneCapture(node,state,description)
    pool=ImmediatePool(binding,owned)
    backend=NS(node=node,state=state,capture=capture,pool=pool,entered=True,volume_active=True)
    return node,state,backend,original_predicates


@pytest.mark.parametrize('distance,count',[(0.,61),(1e-13,61),(.05,61),(.24,61),(.24,17)])
def test_payload_then_adaptive_body_exactly_matches_unchanged_production_method(monkeypatch,distance,count):
    from erc_phase1_solution.manipulation_node import ManipulationNode
    node,state,backend,events=fixture(monkeypatch,count=count)
    first=np.zeros(8);last=first.copy();last[1]=distance
    assert ManipulationNode._carried_volume_transition_is_safe(node,first,last,corners(),
        right_positions=np.zeros(7),head_positions=np.zeros(2))
    expected=copy.deepcopy(events);events.clear()
    before=(state.samples,state.cache_hits,dict(state.cache),state.last_rejection,state.minimum_moving_left_z)
    assert volume.evaluate_volume(backend,first,last,corners())
    assert len(events)==len(expected)
    for (op,q,kw),(old_op,old_q,old_kw) in zip(events,expected):
        assert op==old_op and q.tobytes()==old_q.tobytes() and set(kw)==set(old_kw)
        for key,value in kw.items():np.testing.assert_equal(value,old_kw[key])
    assert before==(state.samples,state.cache_hits,dict(state.cache),state.last_rejection,state.minimum_moving_left_z)


@pytest.mark.parametrize('payload,body',[(.025,None),(None,.025)])
def test_first_failure_and_committed_prefix_match_original_pass_order(monkeypatch,payload,body):
    from erc_phase1_solution.manipulation_node import ManipulationNode
    node,state,backend,events=fixture(monkeypatch,reject_payload=payload,reject_body=body)
    first=np.zeros(8);last=first.copy();last[1]=.05
    assert not ManipulationNode._carried_volume_transition_is_safe(node,first,last,corners(),
        right_positions=np.zeros(7),head_positions=np.zeros(2))
    expected=[(op,q.tobytes()) for op,q,_ in events];events.clear()
    state.last_rejection={'reason':'existing_scene_reason'}
    assert not volume.evaluate_volume(backend,first,last,corners())
    assert expected==[(op,q.tobytes()) for op,q,_ in events]
    assert state.last_rejection=={'reason':'existing_scene_reason'}
    assert backend.last_volume_rejection['reason']==('payload_robot' if payload is not None else 'arm_head')


def test_velocity_or_orientation_leg_rejection_starts_no_queries(monkeypatch):
    node,_,backend,_=fixture(monkeypatch)
    first=np.zeros(8);last=first.copy();last[1]=.26
    assert not volume.evaluate_volume(backend,first,last,corners())
    assert backend.pool.queries==[]
    last[1]=.1;node.chain.pose_error=lambda a,b:np.r_[np.zeros(3),.31,0.,0.]
    assert not volume.evaluate_volume(backend,first,last,corners())
    assert backend.pool.queries==[]


def test_tilt_rejection_precedes_payload_mesh_and_body_pass(monkeypatch):
    from erc_phase1_solution.manipulation_node import ManipulationNode
    node,_,backend,events=fixture(monkeypatch)
    def pose(q):
        value=np.eye(4);value[:3,:3]=np.diag([1.,-1.,-1.]);value[2,3]=1.
        return value
    node.chain.forward=pose
    q=np.zeros(8)
    assert not ManipulationNode._carried_volume_transition_is_safe(node,q,q,corners())
    assert events==[]
    assert not volume.evaluate_volume(backend,q,q,corners()) and events==[]
    assert backend.last_volume_rejection==dict(operation='place_payload',sample=0,reason='place_payload_tilt')
    assert len(backend.pool.queries)==1


@pytest.mark.parametrize('change',['tilt','support','step','orientation','attachment','state_context',
    'head_stale','right_drift','base_drift','cancel'])
def test_every_enqueue_rejects_changed_model_attachment_or_fresh_scene(monkeypatch,change):
    node,state,backend,_=fixture(monkeypatch)
    first=np.zeros(8);last=first.copy();last[1]=.05
    def change_once():
        if len(backend.pool.queries)!=1:return
        if change=='tilt':node.carried_maximum_tilt=.61
        elif change=='support':node.carried_supported_jaw_vertical_component=.71
        elif change=='step':node.cartesian_joint_step=.26
        elif change=='orientation':node.carried_orientation_step_limit=.31
        elif change=='attachment':node._held_book_corners[0,0]+=.001
        elif change=='state_context':state._place_volume_context['aperture']=.018
        elif change=='head_stale':node._joint_stamps_ns['head_1_joint']=0
        elif change=='right_drift':node.joints['arm_right_1_joint']=.002
        elif change=='base_drift':node._staging_odom['pose'][0]=.003
        else:node._cancel.set()
    backend.pool.before_capture=change_once
    with pytest.raises((RuntimeError,ValueError)):
        volume.evaluate_volume(backend,first,last,corners())
    assert len(backend.pool.queries)==1


@pytest.mark.parametrize('change',['epoch','scene_id','model_id','source_id','aperture','repeat','resource'])
def test_worker_rejects_wrong_identity_or_shared_order_and_resource_limit(monkeypatch,change):
    _,_,backend,_=fixture(monkeypatch)
    sample=backend.capture.capture(0,(np.zeros(8),.017,True),pool=backend.pool)
    current=volume.PlaceVolumeQuery(sample,'place_payload')
    if change in ('scene_id','model_id','source_id'):current=replace(current,sample=replace(sample,**{change:'f'*64}))
    elif change=='epoch':current=replace(current,sample=replace(sample,epoch=2))
    elif change=='aperture':current=replace(current,sample=replace(sample,aperture=.018))
    elif change=='repeat':backend.pool.owner._last_request_id=0
    else:backend.pool.owner._queries_seen=20000
    with pytest.raises(RuntimeError):volume.evaluate_owned_volume(backend.pool.owner,current)


def test_default_backend_does_not_enable_volume_or_dispatch_any_pool():
    backend=ProcessPlanningBackend(object(),object(),epoch=1)
    assert not backend.include_carried_volume and not backend.volume_active and backend.pool is None
    with pytest.raises(GeometryProcessError,match='not active'):
        backend.volume(np.zeros(8),np.zeros(8),corners())
    for value in (None,1,'true'):
        with pytest.raises(ValueError):ProcessPlanningBackend(object(),object(),epoch=1,include_carried_volume=value)


def test_exact_owner_admission_rejects_overridden_payload_and_subclasses():
    owner=object.__new__(RobotGeometryOwner)
    assert volume.ordinary_volume_parent(owner)
    owner._carried_robot_collision=lambda *a,**kw:None
    assert not volume.ordinary_volume_parent(owner)
    class Custom(RobotGeometryOwner):pass
    assert not volume.ordinary_volume_parent(object.__new__(Custom))


def test_managed_exit_reaps_even_on_finish_failure_and_returns_no_plan():
    events=[]
    backend=ProcessPlanningBackend(object(),object(),epoch=1,include_carried_volume=True)
    def fail():events.append('finish');raise RuntimeError('finish identity mismatch')
    with pytest.raises(RuntimeError,match='identity mismatch'):
        with backend:
            backend.pool=NS(finish=fail,close=lambda:events.append('close'))
    assert events==['finish','close'] and not backend.entered


def test_managed_cancel_or_planning_failure_closes_without_finish():
    events=[]
    backend=ProcessPlanningBackend(object(),object(),epoch=1,include_carried_volume=True)
    with pytest.raises(RuntimeError,match='cancel'):
        with backend:
            backend.pool=NS(finish=lambda:events.append('finish'),close=lambda:events.append('close'))
            raise RuntimeError('cancel during planning')
    assert events==['close'] and not backend.entered


@pytest.mark.parametrize('selected',[False,True])
def test_actual_scene_planner_retains_default_calls_or_explicit_three_volume_sites(planning,monkeypatch,selected):
    from erc_phase1_solution import scene_checked_place as planner
    from erc_phase1_solution import geometry_process_parent as parent
    node,events,call=planning
    node.carried_transition_samples=61
    attached=[]
    class Managed:
        include_carried_volume=selected
        volume_active=False
        def attach(self,state,description,**kwargs):
            attached.append((description,kwargs));self.volume_active=selected
            return state
        def volume(self,a,b,book):events.append(('volume',a.copy(),b.copy()));return True
    backend=Managed();original=planner.plan_scene_checked_place
    monkeypatch.setattr(parent,'ProcessPlanningBackend',Managed)
    monkeypatch.setattr(planner,'plan_scene_checked_place',lambda *a,**kw:original(*a,geometry_backend=backend,**kw))
    result=call()
    assert result.solutions and len(attached)==1
    assert attached[0][1]==({'volume_aperture':.017} if selected else {})
    assert set(attached[0][0])=={'bin_floor_point','bin_scene','table_scene','attached_corners','transition_samples'}
    route_kwargs=[data for name,data,*rest in events if name=='controller_route']
    assert route_kwargs==[dict(require_gravity_support=True,**({'geometry_backend':backend} if selected else {}))]
    names=[event[0] for event in events]
    expected='volume' if selected else 'loaded'
    assert names[:3]==[expected,'support','scene_leg'] and names.count(expected)==2
    assert not any(name==('loaded' if selected else 'volume') for name in names)
    assert names.index('controller_route')<names.index('opening')


def test_actual_scene_planner_volume_rejection_stops_before_support_scene_or_ik(planning,monkeypatch):
    from erc_phase1_solution import scene_checked_place as planner
    from erc_phase1_solution import geometry_process_parent as parent
    node,events,call=planning;node.carried_transition_samples=61
    class Managed:
        include_carried_volume=True;volume_active=False
        def attach(self,state,description,**kwargs):self.volume_active=True;return state
        def volume(self,*args):events.append(('volume_rejected',));return False
    backend=Managed();original=planner.plan_scene_checked_place
    monkeypatch.setattr(parent,'ProcessPlanningBackend',Managed)
    monkeypatch.setattr(planner,'plan_scene_checked_place',lambda *a,**kw:original(*a,geometry_backend=backend,**kw))
    with pytest.raises(RuntimeError,match='torso_rejected'):call()
    assert events==[('volume_rejected',)]
