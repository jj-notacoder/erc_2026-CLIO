"""Outside-only adapter for an already validated _lift_first_measurements result.

Does not capture sensors, refresh evidence, publish commands or waive subsequent
checks. Invoke only as an explicitly selected geometry experiment.
"""
import math
import numpy as np
from .aperture_interval_geometry import (
    ApertureInterval, ContextEvidence, MASTER, RIGHT_ARM_JOINTS,
    validate_lift_first_route_with_interval, _vector,
)


def certify_captured_route(node,front,lift_plan,bay,measured,*,model,original_interval,
                          assume_nominal_mimics_and_rigid_attachment=False):
    return _certify(node,front,lift_plan,bay,measured,model,original_interval,
                    assume_nominal_mimics_and_rigid_attachment,predicted=False)


def certify_predicted_route(node,front,lift_plan,bay,measured,*,model,original_interval,
                           assume_nominal_mimics_and_rigid_attachment=False):
    """Pre-close geometry: measured open q stays separate from nominal reference."""
    return _certify(node,front,lift_plan,bay,measured,model,original_interval,
                    assume_nominal_mimics_and_rigid_attachment,predicted=True)


def _certify(node,front,lift_plan,bay,measured,model,original_interval,assume,predicted):
    if assume is not True:
        raise ValueError('explicit optional geometry assumptions required')
    if measured.get('finger_geometry_basis') != 'measured_master_with_official_urdf_mimics':
        raise ValueError('only the existing unmeasured nominal-mimic mode is supported')
    if measured.get('reported_passive_joints') != []:
        raise ValueError('any reported/partial passive inventory requires another model')
    observed_master = measured['fingers'][MASTER]
    if not isinstance(original_interval,ApertureInterval):
        raise ValueError('original geometry interval required')
    reference = original_interval.reference_m
    if not predicted and reference != observed_master:
        raise ValueError('original geometry reference differs from captured measured master')
    evidence = ContextEvidence(capture_clock_ns=measured['stamp_ns'],base_odom=tuple(measured['base_pose']),
        source='Existing _lift_first_measurements: checked onboard joint states and stationary odometry; stamp is capture clock, not each producer stamp',
        observed_master_m=observed_master,reference_basis='predicted_nominal_master' if predicted else 'measured_master',
        reported_passive_joints=(),unmeasured_nominal_mimics_assumed=True,rigid_attachment_assumed=True)
    return validate_lift_first_route_with_interval(node,front,measured['left'],lift_plan.route,bay=bay,
        aperture=reference,finger_positions={MASTER:reference},attached_corners=lift_plan.attached_corners,
        lift_m=node.lift_first_extraction_lift_m,modeled_tool_allowance_m=measured['modeled_tool_allowance_m'],
        right_positions=tuple(measured['joints'][name] for name in RIGHT_ARM_JOINTS),
        head_positions=tuple(measured['joints'][name] for name in ('head_1_joint','head_2_joint')),
        interval=original_interval,model=model,evidence=evidence)


def check_closed_geometry_context(certificate,current,*,route,attached_corners,asset_binding,policy):
    """Existing engineering drift policy, NOT a mathematical pose-variation bound.

    Caller must supply a freshly validated _lift_first_measurements result and
    retain original-reference, pressure, force/effort, identity and motion gates.
    This function reads supplied data only and cannot publish or clear faults.
    """
    binding = certificate.as_dict()['binding']
    if current.get('reported_passive_joints') != [] or current.get('finger_geometry_basis') != 'measured_master_with_official_urdf_mimics':
        raise ValueError('closed passive availability differs from nominal certificate')
    if set(current['fingers']) != {MASTER}:
        raise ValueError('closed finger dictionary is not master-only')
    certificate.check_snapshot_binding(binding,current['fingers'][MASTER],current['reported_passive_joints'])
    if asset_binding != binding['assets'] or policy != binding['policy']:
        raise ValueError('geometry assets or policy changed')
    if np.asarray(route,dtype=float).tolist() != binding['route'] or np.asarray(attached_corners,dtype=float).tolist() != binding['attached_corners']:
        raise ValueError('route or attachment changed')
    left = _vector(current['left'],8,'current arm')
    expected_left = np.asarray(binding['measured_start'])
    if np.max(np.abs(left-expected_left)) > .0002:
        raise ValueError('arm joint-vector drift exceeds existing admission policy')
    exact = np.array_equal(left,expected_left)
    for names,key in ((RIGHT_ARM_JOINTS,'right_positions'),(('head_1_joint','head_2_joint'),'head_positions')):
        actual = _vector([current['joints'][name] for name in names],len(names),key)
        expected = np.asarray(binding[key])
        if np.max(np.abs(actual-expected)) > .001:
            raise ValueError('right/head drift exceeds existing admission policy')
        exact = exact and np.array_equal(actual,expected)
    base = _vector(current['base_pose'],3,'current base')
    previous = np.asarray(binding['context']['base_odom'])
    yaw = math.atan2(math.sin(base[2]-previous[2]),math.cos(base[2]-previous[2]))
    if np.linalg.norm(base[:2]-previous[:2]) > .002 or abs(yaw) > .005:
        raise ValueError('base drift exceeds existing admission policy')
    exact = exact and np.array_equal(base,previous)
    return dict(geometry_interval_and_engineering_drift_policy_passed=True,exact_arm_context_match=bool(exact),
                drift_is_proved_displacement_envelope=False,pressure_or_freshness_verified=False,
                motion_permit=False,observed_closed_master_m=current['fingers'][MASTER],
                original_geometry_reference_basis=binding['context']['reference_basis'])
