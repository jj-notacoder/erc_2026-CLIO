"""Optional pre-close nominal geometry; no closure or arm controller.

The interval covers the recorded discrete arm states under ideal official mimics.
Later arm drift is admitted only by the existing engineering tolerance, not a
new displacement proof. All sensor/pressure/action gates remain independent.
"""
from dataclasses import dataclass
import math
import hashlib
import importlib
from pathlib import Path

import numpy as np

from .aperture_interval_geometry import ApertureInterval, OfficialNominalModel, MASTER, _policy_values, _canonical
from .aperture_interval_capture import certify_predicted_route, check_closed_geometry_context
from .motion_profiles import IK_JOINTS, RIGHT_ARM_JOINTS


INTERVAL = ApertureInterval(.01825, .0175, .019)


def invalidate(node):
    with node._lock:
        node._preclose_aperture_generation = getattr(node, '_preclose_aperture_generation', 0)+1
        node._preclose_aperture_state = None
        node._preclose_aperture_checked = None
        node._preclose_aperture_dispatched = False


def _hazards(node):
    if node._cancel.is_set(): raise RuntimeError('preclose aperture cancelled')
    reason = (getattr(node, '_payload_hazard_latched', None)
              or getattr(node, '_held_grip_sensor_fault', None)
              or getattr(node, '_adaptive_motion_halt_reason', None)
              # The final caller already holds the non-reentrant sensor lock.
              or getattr(node, '_adaptive_overload_latched', None)
              or ('payload_robot_contact' if getattr(node, '_target_robot_contact_latched', False) else None))
    if reason: raise RuntimeError('preclose aperture hazard: '+str(reason))


def _precision(node, measured, grasp):
    error = node.chain.pose_error(node.chain.forward(measured['left']), node.chain.forward(grasp))
    if (np.linalg.norm(error[:3]) > node.pick_position_tolerance
            or np.linalg.norm(error[3:]) > node.pick_orientation_tolerance
            or abs(measured['left'][0]-grasp[0]) > .0005
            or np.max(np.abs(measured['left'][1:]-np.asarray(grasp)[1:])) > .01):
        raise RuntimeError('preclose measured grasp is outside planned precision')


def _open(node, measured):
    if (not getattr(node, '_gripper_open_confirmed', False)
            or set(measured['fingers']) != {MASTER}
            or abs(measured['fingers'][MASTER]-node.gripper_open) > node.adaptive_start_tolerance):
        raise RuntimeError('preclose interval needs the measured open master')


def _model(node):
    model = getattr(node, '_preclose_aperture_model', None)
    if model is None:
        from ament_index_python.packages import get_package_share_directory
        urdf = Path(get_package_share_directory('erc_description'))/'urdf/tiago_pro.urdf'
        manifest = Path(get_package_share_directory('erc_phase1_solution'))/'config/aperture_interval_official_assets.json'
        model = OfficialNominalModel(urdf, get_package_share_directory, manifest)
        node._preclose_aperture_model = model
    model.verify_assets_unchanged()
    model.verify_node(node)
    return model


def _reference_signature(reference):
    return _canonical(dict(base_pose=np.asarray(reference['base_pose']).tolist(),
        stamp_ns=reference['stamp_ns'], joints={name:float(reference['joints'][name])
        for name in (*IK_JOINTS, *RIGHT_ARM_JOINTS, 'head_1_joint', 'head_2_joint')}))


@dataclass(frozen=True)
class PrecloseState:
    generation: int
    certificate: object
    model: object
    grasp: tuple
    front_source: object
    plan_source: object
    reference_source: object
    original_reference_signature: str
    bay_source: object
    geometry_object: object
    orientation_tolerance: float

    def binding(self): return self.certificate.as_dict()['binding']

    def check_owner(self, node):
        if (getattr(node, '_preclose_aperture_generation', None) != self.generation
                or getattr(node, '_preclose_aperture_state', None) is not self
                or getattr(node, '_preclose_aperture_dispatched', False)):
            raise RuntimeError('preclose aperture certificate invalidated or consumed')
        _hazards(node)

    def check_model_binding(self, node, *, files):
        binding = self.binding()
        if (node._shelf_cradle_geometry is not self.geometry_object
                or self.model.asset_binding != binding['assets']
                or _policy_values(node) != binding['policy']
                or np.asarray(self.front_source).tolist() != binding['front']
                or np.asarray(self.plan_source.route).tolist() != binding['route']
                or np.asarray(self.plan_source.attached_corners).tolist() != binding['attached_corners']):
            raise RuntimeError('preclose aperture geometry binding changed')
        if _reference_signature(self.reference_source) != self.original_reference_signature:
            raise RuntimeError('preclose aperture original reference changed')
        bay = {k: list(v) if isinstance(v, (np.ndarray, tuple)) else v for k, v in vars(self.bay_source).items()}
        if bay != binding['bay']: raise RuntimeError('preclose aperture bay changed')
        if (node.lift_first_extraction_lift_m != binding['lift_m']
                or node.pick_orientation_tolerance != self.orientation_tolerance):
            raise RuntimeError('preclose aperture admission policy changed')
        if files:
            for name, expected in binding['software_sha256'].items():
                path = Path(importlib.import_module('erc_phase1_solution.'+name).__file__)
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise RuntimeError('preclose aperture geometry source changed')
            from . import aperture_interval_geometry
            if hashlib.sha256(Path(aperture_interval_geometry.__file__).read_bytes()).hexdigest() != binding['interval_helper_sha256']:
                raise RuntimeError('preclose aperture interval source changed')
            self.model.verify_assets_unchanged()
            self.model.verify_node(node)

    def check_current(self, node, current):
        self.check_owner(node)
        self.check_model_binding(node, files=False)
        binding = self.binding()
        if current['stamp_ns'] < binding['context']['capture_clock_ns']:
            raise RuntimeError('preclose aperture clock reversed')
        _precision(node, current, self.grasp)
        if current['modeled_tool_allowance_m'] != binding['modeled_tool_allowance_m']:
            raise RuntimeError('preclose aperture modeled allowance changed')
        return check_closed_geometry_context(self.certificate, current,
            route=self.plan_source.route, attached_corners=self.plan_source.attached_corners,
            asset_binding=self.model.asset_binding, policy=_policy_values(node))


def prepare(node, front, grasp, plan, bay, reference):
    """Capture actual open-arm context after approach; certify predicted q only."""
    _hazards(node)
    model = _model(node)  # Asset-only work precedes the fresh capture.
    generation = node._preclose_aperture_generation
    reference_signature = _reference_signature(reference)
    orientation_tolerance = node.pick_orientation_tolerance
    measured = node._lift_first_measurements(reference)
    _precision(node, measured, grasp)
    _open(node, measured)
    if measured['reported_passive_joints'] or measured['finger_geometry_basis'] != 'measured_master_with_official_urdf_mimics':
        raise RuntimeError('preclose aperture supports only explicitly unmeasured official mimics')
    node._publish_status('preclose_aperture_geometry_started', command='pick',
        observed_open_master=measured['fingers'][MASTER], predicted_reference=INTERVAL.reference_m,
        original_interval=[INTERVAL.lower_m, INTERVAL.upper_m], measured_start=measured['left'].tolist())
    certificate = certify_predicted_route(node, front, plan, bay, measured, model=model,
        original_interval=INTERVAL, assume_nominal_mimics_and_rigid_attachment=True)
    current = node._lift_first_measurements(reference)
    _hazards(node); _precision(node, current, grasp); _open(node, current)
    binding = certificate.as_dict()['binding']
    context = dict(right_positions=tuple(binding['right_positions']), head_positions=tuple(binding['head_positions']),
                   stamp_ns=binding['context']['capture_clock_ns'])
    node._check_lift_collision_context(current, context)
    if (current['stamp_ns'] < measured['stamp_ns']
            or current['reported_passive_joints'] != measured['reported_passive_joints']
            or current['finger_geometry_basis'] != measured['finger_geometry_basis']
            or np.max(np.abs(current['left']-measured['left'])) > .0002
            or abs(current['fingers'][MASTER]-measured['fingers'][MASTER]) > .00001):
        raise RuntimeError('preclose aperture context moved during geometry')
    base_delta = np.asarray(current['base_pose'])-np.asarray(measured['base_pose'])
    if np.linalg.norm(base_delta[:2]) > .002 or abs(math.atan2(math.sin(base_delta[2]), math.cos(base_delta[2]))) > .005:
        raise RuntimeError('preclose aperture base moved during geometry')
    if _reference_signature(reference) != reference_signature:
        raise RuntimeError('preclose aperture original reference changed during geometry')
    state = PrecloseState(generation, certificate, model, tuple(grasp), front, plan, reference,
                          reference_signature, bay, node._shelf_cradle_geometry,
                          orientation_tolerance)
    state.check_model_binding(node, files=True)
    with node._lock:
        _hazards(node)
        if node._preclose_aperture_generation != generation:
            raise RuntimeError('preclose aperture state reset during geometry')
        node._preclose_aperture_state = state
        node._preclose_aperture_checked = None
    node._publish_status('preclose_aperture_geometry_verified', command='pick',
        binding_sha256=certificate.binding_sha256, observed_open_master=measured['fingers'][MASTER],
        reference_basis='predicted_nominal_master', geometry_context=context,
        exact_arm_context_match=bool(np.array_equal(current['left'], measured['left'])),
        drift_is_proved_displacement_envelope=False)
    return state


def reuse(node, front, grasp, plan, bay, reference):
    """Fresh closed geometry admission; never recalculate a missing certificate."""
    state = getattr(node, '_preclose_aperture_state', None)
    if not isinstance(state, PrecloseState): raise RuntimeError('preclose aperture certificate absent')
    state.check_owner(node)
    if (state.front_source is not front or state.plan_source is not plan
            or state.bay_source is not bay or state.reference_source is not reference
            or tuple(grasp) != state.grasp or getattr(node, '_preclose_aperture_checked', None) is not None):
        raise RuntimeError('preclose aperture attempt binding changed or reused')
    state.check_model_binding(node, files=True)
    current = node._lift_first_measurements(reference)
    evidence = state.check_current(node, current)
    binding = state.binding()
    current['geometry_context'] = dict(right_positions=tuple(binding['right_positions']),
        head_positions=tuple(binding['head_positions']), stamp_ns=binding['context']['capture_clock_ns'])
    current['preclose_interval_admission'] = FinalIntervalAdmission(state)
    with node._lock:
        state.check_owner(node)
        node._preclose_aperture_checked = state
    node._publish_status('preclose_aperture_geometry_reused', command='pick',
        binding_sha256=state.certificate.binding_sha256,
        grasp_joints=list(state.grasp), measured_closed_start=current['left'].tolist(),
        original_measured_start=binding['measured_start'],
        original_interval=[binding['aperture_interval']['lower_m'],binding['aperture_interval']['upper_m']],
        predicted_reference=binding['aperture_interval']['reference_m'],
        observed_closed_master=current['fingers'][MASTER], **evidence)
    return current


def check_after_probe(node, checked_hold, current):
    admission = checked_hold.get('preclose_interval_admission')
    if not isinstance(admission, FinalIntervalAdmission):
        raise RuntimeError('preclose aperture final admission absent')
    admission.state.check_model_binding(node, files=True)
    return admission.state.check_current(node, current)


class FinalIntervalAdmission:
    """Small extra checks inside the unchanged final pressure lock admission."""
    def __init__(self, state): self.state = state

    def check(self, node, snapshot, now):
        state = self.state
        state.check_owner(node)
        if getattr(node, '_preclose_aperture_checked', None) is not state:
            raise RuntimeError('preclose aperture reuse was not admitted')
        state.check_model_binding(node, files=False)
        binding = state.binding()
        if np.asarray(node._held_book_corners).tolist() != binding['attached_corners']:
            raise RuntimeError('preclose aperture held attachment changed')
        passive = [name for name in node._shelf_cradle_geometry.mimics if name.startswith('gripper_left_')
                   and name in snapshot['joints']]
        if passive: raise RuntimeError('preclose aperture passive availability changed')
        current = dict(left=np.asarray([snapshot['joints'][name] for name in IK_JOINTS]),
            joints=snapshot['joints'], fingers={MASTER:snapshot['joints'][MASTER]},
            base_pose=snapshot['odom']['pose'], stamp_ns=now, reported_passive_joints=[],
            finger_geometry_basis='measured_master_with_official_urdf_mimics',
            modeled_tool_allowance_m=.005)
        state.check_current(node, current)
        original_interval = ApertureInterval(**binding['aperture_interval'])
        if not original_interval.contains(snapshot['feedback'][-1].position):
            raise RuntimeError('preclose aperture feedback is outside original interval')

    def consume(self, node, snapshot, now):
        self.check(node, snapshot, now)
        node._preclose_aperture_dispatched = True
