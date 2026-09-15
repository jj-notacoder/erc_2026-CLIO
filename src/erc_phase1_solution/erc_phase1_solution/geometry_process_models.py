"""Bind actual parent geometry to the ordinary model rebuilt by each child."""
import hashlib
import json
import math

import numpy as np

from .kinematics import CollisionMesh, Joint, URDFChain


def _array(value):
    if type(value) is not np.ndarray or value.dtype != np.dtype(np.float64) or not np.isfinite(value).all():
        raise ValueError('ordinary finite geometry array required')
    return [list(value.shape),hashlib.sha256(value.tobytes(order='C')).hexdigest()]


def _chain(chain):
    if (type(chain) is not URDFChain
            or getattr(chain.forward,'__func__',None) is not URDFChain.forward
            or getattr(chain.link_transforms,'__func__',None) is not URDFChain.link_transforms):
        raise ValueError('ordinary installed geometry chain required')
    joints=[]
    for item in chain.joints:
        if type(item) is not Joint or not all(math.isfinite(x) for x in (item.lower,item.upper)):
            raise ValueError('ordinary finite geometry joint required')
        joints.append([item.name,item.parent,item.child,item.kind,
            _array(item.origin),_array(item.axis),float(item.lower),float(item.upper)])
    return dict(joints=joints,active=list(chain.active_names),index=dict(chain._active_index),
                lower=_array(chain.lower),upper=_array(chain.upper))


def _meshes(meshes):
    values=[]
    for mesh in meshes:
        if type(mesh) is not CollisionMesh or type(mesh.watertight) is not bool:
            raise ValueError('ordinary collision mesh required')
        values.append([mesh.link,_array(mesh.triangles),_array(mesh.bounds),mesh.watertight])
    return values


def ordinary_scene_components(scene):
    from .scene_checked_place import PlaceSceneChecker, NominalBinObstacle
    from .shelf_cradle_geometry import ShelfCradleGeometry
    from .table_scene import TableSceneObstacle
    from .empty_pickup_collision import ScreenEnvelope
    def ordinary(value,kind,names):
        return type(value) is kind and all(getattr(getattr(value,name,None),'__func__',None)
                                          is getattr(kind,name) for name in names)
    return (ordinary(scene,PlaceSceneChecker,('sample','leg'))
        and ordinary(scene.obstacle,NominalBinObstacle,('intersects','book_intersects'))
        and ordinary(scene.table,TableSceneObstacle,('intersects','intersects_box'))
        and ordinary(scene.screen,ScreenEnvelope,('world',))
        and ordinary(scene.tool,ShelfCradleGeometry,('local_surfaces','model_local_surface')))


def scene_model_signature(scene):
    """Hash values actually used by the built-in checker, excluding caches.

    This is a startup/end-of-plan model admission, not a per-sample cache or
    collision verdict. Parent and child values must agree exactly. Private
    caches are derived state and do not authorize a different model.
    """
    if not ordinary_scene_components(scene):
        raise ValueError('process geometry requires ordinary registered scene components')
    node,tool,obstacle,table,screen=scene.node,scene.tool,scene.obstacle,scene.table,scene.screen
    values=dict(chains=[_chain(node.chain),_chain(node.right_chain),_chain(node.head_chain)],
        meshes=_meshes(node.carried_collision_meshes),
        tool=dict(meshes=_meshes(tool.model.meshes),grasp=_chain(tool.grasp_chain),
            chains=[[name,_chain(chain)] for name,chain in tool.chains.items()],
            mimics=[[name,dict(element.attrib)] for name,element in tool.mimics.items()],
            watertight=list(tool.watertight.items())),
        screen=dict(chain=_chain(screen.chain),corners=_array(screen.corners)),
        obstacle=dict(point=_array(obstacle.point),rotation=_array(obstacle.rotation),
            origin=_array(obstacle.origin),bounds=_array(obstacle.bounds),margin=obstacle.margin,
            materials=[_array(value) for value in obstacle.material_bounds],
            corners=[_array(value) for value in obstacle.material_corners],
            triangles=[_array(value) for value in obstacle.material_triangles]),
        table=dict(origin=_array(table.origin),rotation=_array(table.rotation),margin=table.margin,
            solids=[[name,_array(bounds),_array(corners)] for name,bounds,corners in table.solids],
            triangles=[_array(value) for value in table.solid_triangles]),
        attached=_array(scene.attached),transition_samples=int(node.carried_transition_samples))
    context=getattr(scene,'_place_volume_context',None)
    if context is not None:
        from .place_volume_geometry import validate_volume_context, capture_volume_context
        current=validate_volume_context(context)
        if capture_volume_context(node,current['aperture']) != current:
            raise ValueError('PLACE carried-volume model parameters changed')
        values['carried_volume_context']=current
    return hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
