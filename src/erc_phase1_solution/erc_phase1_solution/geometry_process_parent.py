"""Optional PLACE adapter; all live measurements and action guards stay here.

Use ProcessPlanningBackend as a context manager and pass it explicitly to
plan_scene_checked_place. No launch flag or node default enables this module.
The caller retains its original post-plan and before-command admissions.
"""
from contextlib import nullcontext
import copy
import math
from types import SimpleNamespace

import numpy as np

from .geometry_process_models import scene_model_signature,ordinary_scene_components
from .geometry_process_pool import GeometryProcessPool, GeometryProcessError
from .geometry_process_protocol import commit_sample, sample_key
from .motion_profiles import RIGHT_ARM_JOINTS
from .placement_scene_context import measured_scene_context
from .pure_geometry_owner import GeometryQuery, HEAD_JOINTS


class PairedSceneCapture:
    """Validate the original parked-scene predicate on one owned sensor copy.

    The scene reference/model objects belong to this planning invocation.
    Left/torso/aperture query values are planned samples. Right/head values and
    their producer freshness are admitted anew for every enqueue, including
    samples that later become parent success-cache hits.
    """
    def __init__(self,node,state,description):
        self.node,self.state=node,state
        self.reference=getattr(node,'_selected_place_scene_reference',None)
        self.bin=getattr(node,'_selected_place_bin_scene',None)
        self.table=getattr(node,'_selected_place_table_scene',None)
        if not all(type(value) is dict for value in (self.reference,self.bin,self.table)):
            raise GeometryProcessError('registered PLACE scene references required')
        self.reference_copy=copy.deepcopy(self.reference)
        self.bin_copy=copy.deepcopy(self.bin);self.table_copy=copy.deepcopy(self.table)
        if (self.bin_copy!=description['bin_scene'] or self.table_copy!=description['table_scene']
                or not np.array_equal(np.asarray(description['attached_corners']),state.attached)):
            raise GeometryProcessError('planning description does not match selected scene')
        self.models=node.carried_collision_meshes
        self.chains=(node.chain,node.right_chain,node.head_chain)
        self.held=node._held_book_corners
        self.held_bytes=np.asarray(self.held,dtype=np.float64).tobytes()
        self.transition_samples=int(node.carried_transition_samples)
        self.right_names=tuple(node.right_chain.active_names)
        self.head_names=tuple(node.head_chain.active_names)
        self.last_admission=None
        self.volume_context=copy.deepcopy(description.get('carried_volume_context'))

    def verify_bindings(self):
        node=self.node
        if (getattr(node,'_selected_place_scene_reference',None) is not self.reference
                or getattr(node,'_selected_place_bin_scene',None) is not self.bin
                or getattr(node,'_selected_place_table_scene',None) is not self.table
                or node.carried_collision_meshes is not self.models
                or any(current is not original for current,original in zip(
                    (node.chain,node.right_chain,node.head_chain),self.chains))
                or node._held_book_corners is not self.held
                or np.asarray(self.held).shape!=(8,3)
                or np.asarray(self.held,dtype=np.float64).tobytes()!=self.held_bytes
                or node.carried_transition_samples!=self.transition_samples
                or self.reference!=self.reference_copy or self.bin!=self.bin_copy or self.table!=self.table_copy):
            raise GeometryProcessError('placement scene/model generation changed')
        if self.volume_context is not None:
            from .place_volume_geometry import capture_volume_context, ordinary_volume_parent
            if (capture_volume_context(node,self.volume_context['aperture']) != self.volume_context
                    or getattr(self.state,'_place_volume_context',None) != self.volume_context
                    or not ordinary_volume_parent(node)):
                raise GeometryProcessError('PLACE carried-volume context changed')

    def capture(self,request_id,item,*,pool):
        node=self.node
        if node._cancel.is_set():
            raise GeometryProcessError('geometry evaluation cancelled')
        q,aperture,loaded=item
        # Exactly one sensor lock; no IPC, hashing of files, or nested existing
        # context method executes while callbacks need this lock.
        with node._lock:
            self.verify_bindings()
            joints=dict(node.joints)
            stamps=dict(getattr(node,'_joint_stamps_ns',{}))
            odom=copy.deepcopy(getattr(node,'_staging_odom',None) or {})
        now=int(node.get_clock().now().nanoseconds)
        clock=SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=now))
        view=SimpleNamespace(_lock=nullcontext(),joints=joints,_joint_stamps_ns=stamps,
            _staging_odom=odom,get_clock=lambda:clock,
            right_chain=SimpleNamespace(active_names=self.right_names),
            head_chain=SimpleNamespace(active_names=self.head_names))
        context=measured_scene_context(view,self.reference_copy)
        right=[context['parked_joints'][name] for name in RIGHT_ARM_JOINTS]
        head=[context['parked_joints'][name] for name in HEAD_JOINTS]
        result=GeometryQuery.capture(request_id=request_id,epoch=pool.epoch,
            source_id=pool.identity.source_id,model_id=pool.identity.model_id,scene_id=pool.scene_id,
            q=q,right=right,head=head,aperture=aperture,loaded=loaded)
        self.last_admission=dict(request_id=request_id,input_sha256=result.input_sha256,
            stamp_ns=now,odom_stamp_ns=odom.get('stamp_ns'),
            producer_stamps_ns={name:stamps[name] for name in (*RIGHT_ARM_JOINTS,*HEAD_JOINTS)})
        return result


class ProcessPlaceChecker:
    """Same ordered samples and diagnostics, with up to four evaluations ahead."""
    def __init__(self,state,pool,capture):
        self.state,self.pool,self.capture=state,pool,capture

    def __getattr__(self,name):
        return getattr(self.state,name)

    def _sequence(self,items):
        return self.pool.evaluate_sequence(items,
            capture=lambda index,item:self.capture.capture(index,item,pool=self.pool),
            is_cached=lambda query:sample_key(query) in self.state.cache,
            consume=lambda query,delta:commit_sample(self.state,query,delta,
                cancelled=self.state.node._cancel.is_set))

    def sample(self,q,aperture,loaded):
        return self._sequence([(q,aperture,loaded)])

    def leg(self,first,last,aperture,loaded):
        first,last=np.asarray(first),np.asarray(last)
        count=(1 if np.array_equal(first,last) else max(
            3,int(self.node.carried_transition_samples),
            int(math.ceil(float(np.max(np.abs(last-first)))/.02))+1))
        fractions=np.linspace(0.,1.,count)
        consumed=[0]
        def commit(query,delta):
            accepted=commit_sample(self.state,query,delta,cancelled=self.node._cancel.is_set)
            if not accepted:
                self.state.last_rejection['fraction']=float(fractions[consumed[0]])
            consumed[0]+=1
            return accepted
        return self.pool.evaluate_sequence(
            ((first+(last-first)*fraction,aperture,loaded) for fraction in fractions),
            capture=lambda index,item:self.capture.capture(index,item,pool=self.pool),
            is_cached=lambda query:sample_key(query) in self.state.cache,consume=commit)


class ProcessPlanningBackend:
    """Explicit opt-in lifetime boundary; the default planner remains local."""
    def __init__(self,node,identity,*,epoch,include_carried_volume=False,**limits):
        if type(include_carried_volume) is not bool:
            raise ValueError('include_carried_volume must be bool')
        self.include_carried_volume=include_carried_volume
        self.volume_active=False
        self.volume_fallback_reason=None
        self.last_volume_rejection=None
        self.node,self.identity,self.epoch,self.limits=node,identity,epoch,limits
        self.entered=False;self.pool=None;self.capture=None;self.state=None
        self.attached=False
        self.fallback_reason=None

    def __enter__(self):
        if self.entered or self.attached:
            raise GeometryProcessError('geometry backend cannot be reused or reentered')
        self.entered=True
        return self

    def attach(self,state,description,*,volume_aperture=None):
        if not self.entered or self.attached:
            raise GeometryProcessError('geometry backend requires one managed planning invocation')
        self.attached=True
        from .exact_world_geometry import ordinary_model_producer
        if (not ordinary_scene_components(state) or state.node is not self.node
                or not ordinary_model_producer(self.node)
                or description['bin_scene'] is None or description['table_scene'] is None):
            self.fallback_reason='unsupported custom or unregistered geometry'
            return state
        if self.include_carried_volume:
            from .place_volume_geometry import capture_volume_context, ordinary_volume_parent
            if ordinary_volume_parent(self.node):
                description=copy.deepcopy(description)
                context=capture_volume_context(self.node,volume_aperture)
                description['carried_volume_context']=context
                state._place_volume_context=copy.deepcopy(context)
                self.volume_active=True
            else:
                self.volume_fallback_reason='unsupported custom volume predicates'
        self.state=state
        self.identity.verify()
        self.capture=PairedSceneCapture(self.node,state,description)
        self.geometry_id=scene_model_signature(state)
        self.pool=GeometryProcessPool(self.identity,epoch=self.epoch,scene=description,
            geometry_id=self.geometry_id,cancelled=self.node._cancel.is_set,**self.limits)
        return ProcessPlaceChecker(state,self.pool,self.capture)

    def volume(self,start,end,attached_corners):
        from .place_volume_geometry import evaluate_volume
        return evaluate_volume(self,start,end,attached_corners)

    def __exit__(self,exception_type,exception,traceback):
        try:
            if self.pool is not None:
                try:
                    if exception_type is None:
                        self.pool.finish()
                finally:
                    self.pool.close()
            if exception_type is None and self.state is not None:
                self.capture.verify_bindings()
                measured_scene_context(self.node,self.capture.reference_copy)
                self.identity.verify()
                if scene_model_signature(self.state)!=self.geometry_id:
                    raise GeometryProcessError('parent geometry changed during planning')
        finally:
            self.entered=False


def plan_place_with_processes(node,*args,identity,epoch,pool_limits=None,**kwargs):
    """Return only after worker barriers/reaping and parent end admissions pass.

    This explicit opt-in API leaves the normal node call and every later action
    guard unchanged. A failed managed exit cannot leak a returned usable plan.
    """
    from .scene_checked_place import plan_scene_checked_place
    if 'geometry_backend' in kwargs:
        raise ValueError('geometry backend is owned by this managed call')
    with ProcessPlanningBackend(node,identity,epoch=epoch,**(pool_limits or {})) as backend:
        result=plan_scene_checked_place(node,*args,geometry_backend=backend,**kwargs)
    return result
