"""Model binding reacts to actual geometry changes, independent of caches."""
from types import SimpleNamespace
import copy

import numpy as np
import pytest

from erc_phase1_solution.geometry_process_models import scene_model_signature
from erc_phase1_solution.kinematics import Joint,URDFChain,CollisionMesh,_box_triangles
from erc_phase1_solution.scene_checked_place import PlaceSceneChecker,NominalBinObstacle
from erc_phase1_solution.shelf_cradle_geometry import ShelfCradleGeometry
from erc_phase1_solution.table_scene import TableSceneObstacle
from erc_phase1_solution.empty_pickup_collision import ScreenEnvelope


def model():
    joint=Joint('joint','base','tip','prismatic',np.eye(4),np.array([0.,0.,1.]),0.,1.)
    chain=URDFChain([joint],['joint'])
    triangles=_box_triangles([.1,.1,.1]);bounds=np.array([triangles.min(axis=(0,1)),triangles.max(axis=(0,1))])
    mesh=CollisionMesh('link',triangles,bounds,True)
    node=SimpleNamespace(chain=chain,right_chain=copy.deepcopy(chain),head_chain=copy.deepcopy(chain),
        carried_collision_meshes=(mesh,),carried_transition_samples=3,joints={})
    tool=ShelfCradleGeometry.__new__(ShelfCradleGeometry)
    tool.model=SimpleNamespace(meshes=(copy.deepcopy(mesh),));tool.grasp_chain=copy.deepcopy(chain)
    tool.chains={'link':copy.deepcopy(chain)};tool.mimics={};tool.watertight={'link':True}
    screen=ScreenEnvelope.__new__(ScreenEnvelope);screen.chain=copy.deepcopy(chain);screen.corners=np.zeros((8,3))
    obstacle=NominalBinObstacle.__new__(NominalBinObstacle)
    obstacle.point=np.zeros(3);obstacle.rotation=np.eye(3);obstacle.origin=np.zeros(3)
    obstacle.bounds=bounds.copy();obstacle.margin=.005;obstacle.material_bounds=[bounds.copy()]
    obstacle.material_corners=[np.zeros((8,3))];obstacle.material_triangles=[triangles.copy()]
    table=TableSceneObstacle.__new__(TableSceneObstacle)
    table.origin=np.zeros(3);table.rotation=np.eye(3);table.margin=.025
    table.solids=[('top',bounds.copy(),np.zeros((8,3)))];table.solid_triangles=[triangles.copy()]
    scene=PlaceSceneChecker.__new__(PlaceSceneChecker)
    scene.node=node;scene.tool=tool;scene.screen=screen;scene.obstacle=obstacle;scene.table=table
    scene.attached=np.zeros((8,3))
    return scene


@pytest.mark.parametrize('field',['chain','mesh','tool','screen','bin','table','attached','sampling'])
def test_changed_actual_model_cannot_keep_parent_child_binding(field):
    scene=model();original=scene_model_signature(scene)
    if field=='chain':scene.node.chain.joints[0].origin[0,3]=.001
    elif field=='mesh':scene.node.carried_collision_meshes[0].triangles[0,0,0]+=.001
    elif field=='tool':scene.tool.model.meshes[0].triangles[0,0,0]+=.001
    elif field=='screen':scene.screen.corners[0,0]=.001
    elif field=='bin':scene.obstacle.material_bounds[0][0,0]-=.001
    elif field=='table':scene.table.margin=.026
    elif field=='attached':scene.attached[0,0]=.001
    else:scene.node.carried_transition_samples=4
    assert scene_model_signature(scene)!=original


def test_model_digest_matches_independent_copy_and_ignores_derived_cache_history():
    scene=model();replica=copy.deepcopy(scene)
    expected=scene_model_signature(scene)
    assert scene_model_signature(replica)==expected
    scene.cache={'old':True};scene.last_rejection={'reason':'past'};scene.samples=99
    scene.node._self_collision_cache={b'old':('a','b')};scene.table.last_intersection='top'
    scene.tool._local_cache={'old':'derived'};scene.node.joints={'measured':.1}
    assert scene_model_signature(scene)==expected


@pytest.mark.parametrize('kind',['chain_override','chain_subclass','nonfinite_mesh','custom_table','tool_override'])
def test_custom_or_nonfinite_model_cannot_enter_ordinary_process_binding(kind):
    scene=model()
    if kind=='chain_override':scene.node.chain.forward=lambda q:np.eye(4)
    elif kind=='chain_subclass':
        class Custom(URDFChain):pass
        scene.node.chain=Custom(scene.node.chain.joints,['joint'])
    elif kind=='nonfinite_mesh':scene.node.carried_collision_meshes[0].triangles[0,0,0]=float('nan')
    elif kind=='custom_table':scene.table=SimpleNamespace()
    else:scene.tool.local_surfaces=lambda aperture:{}
    with pytest.raises(ValueError):scene_model_signature(scene)
