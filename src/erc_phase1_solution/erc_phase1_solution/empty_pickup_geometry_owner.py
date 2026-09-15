"""Pure empty-PICK samples in the existing bounded geometry worker protocol.

One composed ordinary robot owner and the original empty checker evaluate the
same nominal samples. No held book, PLACE scene, motion or sensor admission is
invented here. The parent retains grid order, cache commits and live guards.
"""
from dataclasses import dataclass
import json
import math

import numpy as np

from .pure_geometry_owner import GeometryQuery, RobotGeometryOwner, _canonical_digest
from .geometry_process_protocol import SampleDelta

SCENE_KIND='empty_pickup_v1'


def _array(value,count,label):
    current=np.asarray(value,dtype=np.float64)
    if current.shape!=(count,) or not np.isfinite(current).all():
        raise ValueError('invalid empty pickup '+label)
    return np.frombuffer(current.tobytes(),dtype=np.float64)


def validate_empty_pickup_scene(scene):
    fields={'kind','start','right','head','initial_aperture','open_aperture','transition_samples'}
    if type(scene) is not dict or set(scene)!=fields or scene.get('kind')!=SCENE_KIND:
        raise ValueError('invalid empty pickup scene fields')
    current=json.loads(json.dumps(scene,sort_keys=True,allow_nan=False))
    for name,count in (('start',8),('right',7),('head',2)):_array(current[name],count,name)
    for name,lower,upper in (('initial_aperture',-1e-6,.069+1e-6),('open_aperture',0.,.069)):
        value=current[name]
        if type(value) not in (int,float) or not math.isfinite(value) or not lower<=value<=upper:
            raise ValueError('invalid empty pickup '+name)
    if type(current['transition_samples']) is not int or not 3<=current['transition_samples']<=1000:
        raise ValueError('invalid empty pickup transition count')
    return current


@dataclass(frozen=True)
class EmptyPickupGeometryQuery:
    sample: GeometryQuery
    operation: str='empty_pickup'

    def __getattr__(self,name):return getattr(self.sample,name)

    @property
    def input_sha256(self):
        return _canonical_digest(dict(sample=self.sample.input_sha256,operation=self.operation))

    def validated(self):
        if type(self) is not EmptyPickupGeometryQuery or type(self.operation) is not str or self.operation!='empty_pickup':
            raise ValueError('invalid empty pickup operation')
        if type(self.sample) is not GeometryQuery:
            raise ValueError('empty pickup requires one ordinary geometry query')
        self.sample.validated()
        if self.sample.loaded is not False:
            raise ValueError('empty pickup query cannot carry a payload')
        return self


def empty_model_signature(checker,scene):
    """Bind the ordinary actual checker, full fixed context and geometry values."""
    from .empty_pickup_collision import EmptyPickupCollision,ScreenEnvelope
    from .shelf_cradle_geometry import ShelfCradleGeometry
    from .exact_world_geometry import ordinary_model_producer
    from .geometry_process_models import _array as model_array,_chain,_meshes
    current=validate_empty_pickup_scene(scene)
    def ordinary(value,kind,names):
        return type(value) is kind and all(getattr(getattr(value,name,None),'__func__',None) is getattr(kind,name) for name in names)
    if not ordinary(checker,EmptyPickupCollision,('sample','_sample_uncached','_cached_robot_world')):
        raise ValueError('ordinary empty pickup checker required')
    node,tool,screen=checker.node,checker.geometry,checker.screen
    if (not ordinary_model_producer(node)
            or not ordinary(tool,ShelfCradleGeometry,('local_surfaces','model_local_surface'))
            or not ordinary(screen,ScreenEnvelope,('world',))):
        raise ValueError('ordinary empty pickup geometry required')
    for name,count in (('start',8),('right',7),('head',2)):
        if _array(getattr(checker,name),count,name).tobytes()!=_array(current[name],count,name).tobytes():
            raise ValueError('empty pickup checker context differs from scene: '+name)
    for name in ('initial_aperture','open_aperture'):
        if np.float64(getattr(checker,name)).tobytes()!=np.float64(current[name]).tobytes():
            raise ValueError('empty pickup checker aperture differs from scene')
    if (np.float64(node.gripper_open).tobytes()!=np.float64(current['open_aperture']).tobytes()
            or type(node.carried_transition_samples) is not int
            or node.carried_transition_samples!=current['transition_samples']):
        raise ValueError('empty pickup sampling policy differs from scene')
    if (set(checker.context)!={'right_positions','head_positions'}
            or _array(checker.context['right_positions'],7,'right context').tobytes()!=_array(current['right'],7,'right').tobytes()
            or _array(checker.context['head_positions'],2,'head context').tobytes()!=_array(current['head'],2,'head').tobytes()):
        raise ValueError('empty pickup effective context differs from scene')
    values=dict(scene=current,
        chains=[_chain(node.chain),_chain(node.right_chain),_chain(node.head_chain)],
        meshes=_meshes(node.carried_collision_meshes),
        tool=dict(meshes=_meshes(tool.model.meshes),grasp=_chain(tool.grasp_chain),
            chains=[[name,_chain(chain)] for name,chain in tool.chains.items()],
            mimics=[[name,dict(element.attrib)] for name,element in tool.mimics.items()],
            watertight=list(tool.watertight.items())),
        screen=dict(chain=_chain(screen.chain),corners=model_array(screen.corners)))
    return _canonical_digest(values)


class EmptyPickupGeometryOwner:
    """A composed ordinary model keeps the existing exact world-cache path."""
    maximum_queries=RobotGeometryOwner.maximum_queries
    maximum_distinct_apertures=RobotGeometryOwner.maximum_distinct_apertures

    def __init__(self,assets):
        self.robot=RobotGeometryOwner(assets)
        self.assets=self.robot.assets
        self.checker=None
        self._epoch=-1
        self._last_request_id=-1
        self._queries_seen=0
        self._apertures=set()
        self._empty_scene=None

    def begin_scene(self,*,epoch,scene):
        from .empty_pickup_collision import EmptyPickupCollision,ScreenEnvelope
        if type(epoch) is not int or epoch<=self._epoch:
            raise ValueError('empty pickup scene epoch must advance')
        if self.checker is not None:
            raise RuntimeError('empty pickup owner has an immutable active scene')
        current=validate_empty_pickup_scene(scene)
        self.robot.gripper_open=current['open_aperture']
        self.robot.carried_transition_samples=current['transition_samples']
        checker=EmptyPickupCollision(self.robot,current['start'],current['right'],current['head'],
            current['initial_aperture'],geometry=self.robot._shelf_cradle_geometry,screen=ScreenEnvelope(self.assets.urdf))
        if checker._world_geometry_cache is None:
            raise RuntimeError('ordinary empty owner world cache was not admitted')
        self.checker=checker
        self._empty_scene=current
        self._scene_id=_canonical_digest(current)
        self._epoch=epoch
        return self._scene_id

    def model_signature(self):
        if self.checker is None or _canonical_digest(self._empty_scene)!=self._scene_id:
            raise RuntimeError('empty pickup scene absent or changed')
        return empty_model_signature(self.checker,self._empty_scene)

    def cancel(self):self.robot.cancel()

    def evaluate_independent(self,query):
        if type(query) is not EmptyPickupGeometryQuery:
            raise ValueError('empty owner requires an empty pickup query')
        query.validated()
        if self.checker is None:
            raise RuntimeError('no active empty pickup scene')
        if (query.epoch!=self._epoch or query.source_id!=self.assets.source_id
                or query.model_id!=self.assets.model_id or query.scene_id!=self._scene_id):
            raise RuntimeError('empty pickup query identity mismatch')
        if (query.right!=self.checker.right.tobytes() or query.head!=self.checker.head.tobytes()):
            raise RuntimeError('empty pickup query changes frozen right/head context')
        if query.request_id<=self._last_request_id:
            raise RuntimeError('empty pickup queries must arrive in increasing order')
        if (self._queries_seen>=self.maximum_queries
                or (query.aperture not in self._apertures and len(self._apertures)>=self.maximum_distinct_apertures)):
            self.robot.cancel()
            raise RuntimeError('empty pickup geometry resource bound exceeded')
        self._last_request_id=query.request_id
        self._queries_seen+=1
        self._apertures.add(query.aperture)
        try:
            reason=self.checker.sample(np.frombuffer(query.q,dtype=np.float64),query.aperture)
        except BaseException:
            self.robot.cancel()
            raise
        if self.robot._cancel.is_set():reason='empty_pickup_cancelled'
        if reason is not None and type(reason) is not str:
            raise RuntimeError('empty pickup sample returned an invalid reason')
        return SampleDelta(query.request_id,query.epoch,query.source_id,query.model_id,
            query.scene_id,query.input_sha256,reason is None,None,
            None if reason is None else {'reason':reason},False,None).validate()
