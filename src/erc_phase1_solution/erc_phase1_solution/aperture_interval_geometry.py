"""Outside-production optional nominal aperture-range geometry check.

No ROS imports, controllers, commands or pressure-loop integration. Certificates
describe continuous nominal master variation at the existing discrete arm states.
Unobserved passive deflection and rigid book attachment remain assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence
from types import MappingProxyType
import xml.etree.ElementTree as ET

import numpy as np

from erc_phase1_solution.kinematics import (
    PreparedTriangleMesh, URDFChain, load_urdf_collision_meshes, triangle_meshes_intersect,
)
from erc_phase1_solution import lift_first_extraction as lift
from erc_phase1_solution.rigid_palm_preflight import (
    LEFT_GRIPPER_COLLISION_LINKS, INNER_OUTER_FINGER_COLLISION_LINKS,
    FINGERTIP_COLLISION_LINKS, PALM_COLLISION_LINK,
)
from erc_phase1_solution.shelf_cradle_geometry import ShelfCradleGeometry
from erc_phase1_solution.motion_profiles import IK_JOINTS, RIGHT_ARM_JOINTS

OFFICIAL_URDF_SHA256 = 'a9ab0ba01932706e4ba1539043cdf0533c77aa66b31611509454637e454a76b0'
MASTER = 'gripper_left_finger_joint'
ROBOT_LINKS = ('base_link','torso_base_link','torso_lift_link',
               *(f'arm_{side}_{i}_link' for side in ('left','right') for i in range(1,8)),
               'head_1_link','head_2_link','head_front_camera_link')
NUMERICAL_MARGIN_M = 1e-9  # Existing metric mesh tolerance, never a deflection allowance.


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _vector(value, count, label):
    array = np.asarray(value,dtype=float)
    if array.shape != (count,) or not np.all(np.isfinite(array)):
        raise ValueError(f'{label} needs {count} finite values')
    return np.frombuffer(array.tobytes(),dtype=float)


def _surface(value):
    array = np.asarray(value,dtype=float)
    if array.ndim != 3 or array.shape[1:] != (3,3) or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError('unsupported empty, nonfinite or malformed collision mesh')
    return array


def _mesh_signature(meshes):
    return tuple((m.link,tuple(m.triangles.shape),bool(m.watertight),
                  hashlib.sha256(np.asarray(m.triangles,dtype=float).tobytes()).hexdigest()) for m in meshes)


def _chain_signature(chain):
    return _canonical(dict(active=list(chain.active_names),joints=[dict(
        name=j.name,parent=j.parent,child=j.child,kind=j.kind,origin=j.origin.tolist(),
        axis=j.axis.tolist(),lower=j.lower,upper=j.upper) for j in chain.joints]))


def _policy_values(node):
    names = ('carried_transition_samples','carried_book_padding','carried_shelf_margin',
             'carried_maximum_tilt','carried_orientation_step_limit','carried_supported_jaw_vertical_component',
             'cartesian_joint_step','cartesian_clearance','pick_position_tolerance')
    return dict(dimensions=np.asarray(node.carried_book_dimensions).tolist(),
                **{name:getattr(node,name,None) for name in names})


@dataclass(frozen=True)
class ApertureInterval:
    reference_m: float
    lower_m: float
    upper_m: float

    def __post_init__(self):
        if not all(math.isfinite(q) for q in (self.reference_m,self.lower_m,self.upper_m)):
            raise ValueError('aperture interval must be finite')
        if not 0 <= self.lower_m <= self.reference_m <= self.upper_m <= .069:
            raise ValueError('original reference must lie inside the official master interval')

    @property
    def maximum_deviation_m(self):
        return max(self.reference_m-self.lower_m,self.upper_m-self.reference_m)

    def contains(self, measured_aperture_m):
        return math.isfinite(measured_aperture_m) and self.lower_m <= measured_aperture_m <= self.upper_m


@dataclass(frozen=True)
class ContextEvidence:
    capture_clock_ns: int
    base_odom: tuple[float,float,float]
    source: str
    observed_master_m: float
    reference_basis: str = 'measured_master'
    reported_passive_joints: tuple[str,...] = ()
    unmeasured_nominal_mimics_assumed: bool = False
    rigid_attachment_assumed: bool = False

    def __post_init__(self):
        if isinstance(self.capture_clock_ns,bool) or not isinstance(self.capture_clock_ns,int) or self.capture_clock_ns <= 0:
            raise ValueError('positive context capture clock epoch required')
        object.__setattr__(self,'base_odom',tuple(_vector(self.base_odom,3,'base odometry')))
        object.__setattr__(self,'reported_passive_joints',tuple(self.reported_passive_joints))
        if not isinstance(self.source,str) or not self.source.strip():
            raise ValueError('onboard context provenance required')
        if not math.isfinite(self.observed_master_m) or not 0 <= self.observed_master_m <= .069:
            raise ValueError('actual observed master must be retained separately')
        if self.reference_basis not in ('measured_master','predicted_nominal_master'):
            raise ValueError('unknown aperture-reference evidence basis')
        if self.reported_passive_joints:
            raise ValueError('reported passive availability unsupported by nominal-only interval mode')
        if self.unmeasured_nominal_mimics_assumed is not True or self.rigid_attachment_assumed is not True:
            raise ValueError('explicit nominal mimic and rigid attachment assumptions required')


@dataclass(frozen=True)
class IntervalCertificate:
    binding_json: str
    binding_sha256: str
    result_json: str

    def as_dict(self):
        return dict(binding=json.loads(self.binding_json),binding_sha256=self.binding_sha256,
                    result=json.loads(self.result_json))

    def check_snapshot_binding(self, binding, measured_aperture_m, reported_passive_joints=()):
        """Geometry-only identity check; not a fresh runtime motion permit."""
        if tuple(reported_passive_joints):
            raise ValueError('passive inventory changed or unsupported')
        if _canonical(binding) != self.binding_json:
            raise ValueError('route, assets, context or attachment binding changed')
        interval = ApertureInterval(**binding['aperture_interval'])
        if not interval.contains(measured_aperture_m):
            raise ValueError('measured aperture outside original certified interval')


class OfficialNominalModel:
    """Load pinned files independently and compare the caller's collision model."""
    def __init__(self, urdf: Path, resolve_package: Callable, manifest_path: Path | None = None):
        self.urdf = Path(urdf)
        manifest_path = manifest_path or Path(__file__).with_name('official_asset_manifest.json')
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest['urdf_sha256'] != OFFICIAL_URDF_SHA256 or hashlib.sha256(self.urdf.read_bytes()).hexdigest() != OFFICIAL_URDF_SHA256:
            raise ValueError('official URDF identity mismatch')
        if set(manifest['collision_links']) != set((*ROBOT_LINKS,*LEFT_GRIPPER_COLLISION_LINKS)):
            raise ValueError('asset manifest collision scope mismatch')
        uris = {mesh.get('filename') for link in ET.parse(self.urdf).getroot().findall('link')
                if link.get('name') in manifest['collision_links']
                for mesh in link.findall('collision/geometry/mesh')}
        if uris != set(manifest['mesh_sha256']):
            raise ValueError('collision asset manifest is incomplete or contains unexpected meshes')
        self.asset_files = [(self.urdf,OFFICIAL_URDF_SHA256)]
        for uri,expected in manifest['mesh_sha256'].items():
            package,relative = uri.removeprefix('package://').split('/',1)
            path = Path(resolve_package(package))/relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('official collision asset mismatch: '+uri)
            self.asset_files.append((path,expected))
        self.asset_binding = dict(urdf_sha256=OFFICIAL_URDF_SHA256,
                                 manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                                 mesh_sha256=manifest['mesh_sha256'])
        self.geometry = ShelfCradleGeometry(self.urdf,resolve_package)
        self.robot_meshes = load_urdf_collision_meshes(self.urdf,ROBOT_LINKS,resolve_package)
        self.tool_signature = _mesh_signature(self.geometry.model.meshes)
        self.robot_signature = _mesh_signature(self.robot_meshes)
        self.radii = MappingProxyType(nominal_link_radii(self.geometry.model.meshes))

    def verify_node(self,node):
        if _mesh_signature(node._shelf_cradle_geometry.model.meshes) != self.tool_signature:
            raise ValueError('caller tool collision mesh differs from pinned model')
        if _mesh_signature(node.carried_collision_meshes) != self.robot_signature:
            raise ValueError('caller robot collision mesh differs from pinned model')
        for name,tip,active in (('chain','gripper_left_grasping_link',IK_JOINTS),
                               ('right_chain','gripper_right_grasping_link',('torso_lift_joint',*RIGHT_ARM_JOINTS)),
                               ('head_chain','head_front_camera_link',('torso_lift_joint','head_1_joint','head_2_joint'))):
            actual = getattr(node,name)
            expected = URDFChain.from_urdf(self.urdf,'base_footprint',tip,active)
            if _chain_signature(actual) != _chain_signature(expected):
                raise ValueError('caller kinematic chain differs from pinned model: '+name)
        actual_tool = node._shelf_cradle_geometry
        if set(actual_tool.chains) != set(self.geometry.chains):
            raise ValueError('caller tool chain inventory differs from pinned model')
        if any(_chain_signature(actual_tool.chains[name]) != _chain_signature(chain)
               for name,chain in self.geometry.chains.items()):
            raise ValueError('caller passive tool chain differs from pinned model')
        if _chain_signature(actual_tool.grasp_chain) != _chain_signature(self.geometry.grasp_chain):
            raise ValueError('caller tool grasp frame differs from pinned model')
        if {name:dict(value.attrib) for name,value in actual_tool.mimics.items()} != {name:dict(value.attrib) for name,value in self.geometry.mimics.items()}:
            raise ValueError('caller mimic map differs from pinned model')
        if actual_tool.watertight != self.geometry.watertight:
            raise ValueError('caller tool watertight flags differ from pinned model')
        if any(getattr(node,name,0.) != 0. for name in ('_pick_book_yaw','_initial_book_yaw')):
            raise ValueError('nonzero initial book-yaw extension is outside this frozen geometry scope')

    def verify_assets_unchanged(self):
        if any(hashlib.sha256(path.read_bytes()).hexdigest() != expected for path,expected in self.asset_files):
            raise ValueError('collision assets changed during certification')


def nominal_link_radii(meshes):
    """Official child-link vertices give hinge-z radius; tips translate on50mm arc."""
    if {m.link for m in meshes} != set(LEFT_GRIPPER_COLLISION_LINKS):
        raise ValueError('exact nine-link tool model required')
    result = {}
    for link in LEFT_GRIPPER_COLLISION_LINKS:
        values = [_surface(m.triangles) for m in meshes if m.link == link]
        if link in INNER_OUTER_FINGER_COLLISION_LINKS:
            radius = max(float(np.linalg.norm(x[...,:2],axis=-1).max()) for x in values)
        elif link in FINGERTIP_COLLISION_LINKS:
            radius = .05  # Pinned URDF: +/-8.28q rotations cancel; origin traces50mm arc.
        else:
            radius = 0.
        result[link] = radius
    return result


def vertex_displacement_bounds(radii: Mapping[str,float], interval: ApertureInterval):
    result = {}
    for link,radius in radii.items():
        if link not in LEFT_GRIPPER_COLLISION_LINKS or not math.isfinite(radius) or radius < 0:
            raise ValueError('unsupported linkage radius')
        exact_lipschitz = radius * 8.28 * interval.maximum_deviation_m
        # Outward floating rounding plus a small numerical buffer; no stiffness claim.
        result[link] = float(np.nextafter(exact_lipschitz,math.inf)+1e-12) if exact_lipschitz else 0.
    if set(result) != set(LEFT_GRIPPER_COLLISION_LINKS):
        raise ValueError('all nine linkage bounds required')
    return result


def interval_mesh_pair_is_separated(first,second,maximum_displacement_m,*,first_watertight,second_watertight):
    """Sufficient nominal separation, with exact reference containment preserved."""
    if not math.isfinite(maximum_displacement_m) or maximum_displacement_m < 0:
        raise ValueError('invalid metric displacement bound')
    a = first if isinstance(first,PreparedTriangleMesh) else PreparedTriangleMesh(_surface(first))
    b = second if isinstance(second,PreparedTriangleMesh) else PreparedTriangleMesh(_surface(second))
    margin = maximum_displacement_m + NUMERICAL_MARGIN_M
    # Expanding the OUTER filter is essential; unexpanded skips are unsound here.
    if np.any(a.bounds[1]+margin < b.bounds[0]) or np.any(b.bounds[1]+margin < a.bounds[0]):
        return True
    if triangle_meshes_intersect(a,b,first_watertight=first_watertight,second_watertight=second_watertight):
        return False
    # Current kernel normalizes each SAT axis: tolerance is a metric vertex budget.
    return not triangle_meshes_intersect(a,b,tolerance=margin,first_watertight=False,second_watertight=False)


def _interval_pass(node,first,goals,front,dimensions,bay,aperture,model,context,allowance):
    local = {name:np.frombuffer(_surface(surface).tobytes(),dtype=float).reshape(surface.shape)
             for name,surface in model.geometry.local_surfaces(aperture.reference_m,{MASTER:aperture.reference_m}).items()}
    displacement = vertex_displacement_bounds(model.radii,aperture)
    marker,left,side_min,side_max = lift._bay_values(bay)
    signs = np.asarray([(x,y,z) for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
    book_initial = front+[dimensions[0]/2,0.,0.]+signs*(dimensions/2)
    roof = float(book_initial[:,2].min())+lift._BAY_HEIGHT-bay.roof_uncertainty_m-bay.margin_m
    robot_watertight = node._watertight_collision_links()
    records = []
    previous = first
    for index,goal in enumerate(goals):
        counts = dict(tool=max(3,int(node.carried_transition_samples),int(math.ceil(np.max(np.abs(goal-previous))/.02))+1),
                      bay=max(61,int(node.carried_transition_samples),int(math.ceil(np.max(np.abs(goal[1:]-previous[1:]))/.02))+1))
        if np.array_equal(previous,goal):
            counts['tool'] = 1
        ground_min = roof_min = side_minimum = math.inf
        tested_pairs = 0
        for fraction in np.linspace(0.,1.,counts['tool']):
            lift._check_cancelled(node)
            q = previous+(goal-previous)*fraction
            hand = lift._pose(node.chain,q)
            robot = {name:PreparedTriangleMesh(surface) for name,surface in node._world_collision_surfaces(q,**context).items()}
            if set(robot) != set(ROBOT_LINKS):
                raise ValueError('robot surface inventory changed')
            for name,triangles in local.items():
                world = PreparedTriangleMesh(triangles@hand[:3,:3].T+hand[:3,3])
                ground = float(world.bounds[0,2]-.02-displacement[name])
                if ground < 0:
                    raise ValueError(f'interval tool-ground clearance:{name}:leg{index}')
                ground_min = min(ground_min,ground)
                for robot_name,surface in robot.items():
                    lift._check_cancelled(node)
                    if name == PALM_COLLISION_LINK and robot_name == 'arm_left_7_link':
                        continue  # Preserve the existing adjacent wrist/palm exception exactly.
                    tested_pairs += 1
                    if not interval_mesh_pair_is_separated(world,surface,displacement[name],
                            first_watertight=model.geometry.watertight[name],second_watertight=robot_name in robot_watertight):
                        raise ValueError(f'interval separation unverified:{name}:{robot_name}:leg{index}')
        for fraction in np.linspace(0.,1.,counts['bay']):
            lift._check_cancelled(node)
            hand = lift._pose(node.chain,previous+(goal-previous)*fraction)
            for name,triangles in local.items():
                world = triangles@hand[:3,:3].T+hand[:3,3]
                lateral = (world-marker)@left
                roof_gap = float(roof-world[...,2].max()-allowance-displacement[name])
                side_gap = float(min(lateral.min()-side_min,side_max-lateral.max())-allowance-displacement[name])
                if roof_gap < 0 or side_gap < 0:
                    raise ValueError(f'interval tool roof/side clearance:{name}:leg{index}')
                roof_min,side_minimum = min(roof_min,roof_gap),min(side_minimum,side_gap)
        records.append(dict(leg=index,samples=counts,pairs=tested_pairs,minimum_tool_ground_clearance_m=ground_min,
                            minimum_tool_roof_clearance_m=roof_min,minimum_tool_side_clearance_m=side_minimum))
        previous = goal
    return dict(legs=records,per_link_vertex_bound_m=displacement)


def validate_lift_first_route_with_interval(
    node,front,measured_start,route,*,bay,aperture,lift_m=.005,
    finger_positions=None,attached_corners=None,modeled_tool_allowance_m=0.,right_positions=None,head_positions=None,
    interval: ApertureInterval | None = None,model: OfficialNominalModel | None = None,evidence: ContextEvidence | None = None,
):
    """Optional geometry wrapper; omitted interval is the unchanged existing API."""
    if interval is None:
        return lift.validate_lift_first_route(node,front,measured_start,route,bay=bay,aperture=aperture,lift_m=lift_m,
            finger_positions=finger_positions,attached_corners=attached_corners,modeled_tool_allowance_m=modeled_tool_allowance_m,
            right_positions=right_positions,head_positions=head_positions)
    lift._check_cancelled(node)
    if not isinstance(interval,ApertureInterval) or not isinstance(model,OfficialNominalModel) or not isinstance(evidence,ContextEvidence):
        raise ValueError('explicit interval, pinned model and captured context required')
    if aperture != interval.reference_m or finger_positions != {MASTER:interval.reference_m}:
        raise ValueError('nominal geometry reference must match the original interval reference')
    if evidence.reference_basis == 'measured_master' and evidence.observed_master_m != aperture:
        raise ValueError('predicted aperture cannot be labeled as measured master')
    first = _vector(measured_start,8,'measured start')
    point = _vector(front,3,'book front')
    goals = tuple(_vector(q,8,'route goal') for q in route)
    if len(goals) != 5:
        raise ValueError('this candidate certifies only the existing five-leg initial route')
    context = dict(right_positions=_vector(right_positions,7,'measured right'),head_positions=_vector(head_positions,2,'measured head'))
    attachment = np.asarray(attached_corners,dtype=float)
    if attachment.shape != (8,3) or not np.all(np.isfinite(attachment)):
        raise ValueError('exact preplanned attachment must be supplied')
    attachment = np.frombuffer(attachment.tobytes(),dtype=float).reshape((8,3))
    model.verify_node(node)
    policy = _policy_values(node)
    # Isolate all supplied geometry/context arrays without changing node feedback.
    dimensions = _vector(node.carried_book_dimensions,3,'book dimensions')
    bay = lift.RelativeShelfBay(**{**vars(bay),
        'marker_center_base':tuple(_vector(bay.marker_center_base,3,'bay marker')),
        'inward_axis_base':tuple(_vector(bay.inward_axis_base,3,'bay inward axis'))})
    nominal = model.geometry.local_surfaces(aperture,{MASTER:aperture})
    caller_nominal = node._shelf_cradle_geometry.local_surfaces(aperture,{MASTER:aperture})
    if set(nominal) != set(caller_nominal) or any(nominal[name].tobytes() != caller_nominal[name].tobytes() for name in nominal):
        raise ValueError('caller cached nominal tool surfaces differ from pinned model')
    bay_binding = {name:(list(value) if isinstance(value,(np.ndarray,tuple)) else value) for name,value in vars(bay).items()}
    binding = dict(aperture_interval=vars(interval),assets=model.asset_binding,front=point.tolist(),measured_start=first.tolist(),
        route=[q.tolist() for q in goals],attached_corners=attachment.tolist(),right_positions=context['right_positions'].tolist(),
        head_positions=context['head_positions'].tolist(),context=vars(evidence),bay=bay_binding,lift_m=lift_m,
        modeled_tool_allowance_m=modeled_tool_allowance_m,carried_transition_samples=node.carried_transition_samples,
        carried_book_dimensions=np.asarray(node.carried_book_dimensions).tolist(),carried_book_padding=node.carried_book_padding,
        carried_shelf_margin=node.carried_shelf_margin,policy=policy,
        software_sha256={name:hashlib.sha256(Path(__import__('erc_phase1_solution.'+name,fromlist=['']).__file__).read_bytes()).hexdigest()
                         for name in ('kinematics','shelf_cradle_geometry','lift_first_extraction')},
        interval_helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    binding_json = _canonical(binding)
    reference = lift.validate_lift_first_route(node,point,first,goals,bay=bay,aperture=aperture,lift_m=lift_m,
        finger_positions={MASTER:aperture},attached_corners=attachment,modeled_tool_allowance_m=modeled_tool_allowance_m,**context)
    extra = _interval_pass(node,first,goals,point,dimensions,bay,interval,model,context,modeled_tool_allowance_m)
    model.verify_node(node)
    if _policy_values(node) != policy:
        raise ValueError('geometry policy changed during certification')
    current_nominal = node._shelf_cradle_geometry.local_surfaces(aperture,{MASTER:aperture})
    if set(nominal) != set(current_nominal) or any(nominal[name].tobytes() != current_nominal[name].tobytes() for name in nominal):
        raise ValueError('nominal tool cache changed during certification')
    model.verify_assets_unchanged()
    lift._check_cancelled(node)
    result = dict(reference=reference,interval=extra,nominal_master_interval_at_sampled_arm_states=True,
                  continuous_arm_motion_certified=False,physical_deflection_certified=False,rigid_attachment_verified=False,
                  shelf_floor_front_edge_mesh_certified=False,deferred_carry_or_recovery_certified=False,
                  sensor_freshness_or_motion_admission_certified=False,post_geometry_commands_sent=False)
    return IntervalCertificate(binding_json,hashlib.sha256(binding_json.encode()).hexdigest(),_canonical(result))
