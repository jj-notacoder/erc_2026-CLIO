"""Offline differential benchmark; no ROS nodes, publishers, or motion.

Use historical joint paths only as representative geometry queries. Book size
and robot collision meshes come from --source-root, not the historical model.
This is a kernel benchmark, not a new path or physical-retention certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time
import xml.etree.ElementTree as ET

import numpy as np

from erc_phase1_solution.kinematics import (
    URDFChain, load_urdf_collision_meshes, oriented_box_from_corners,
    oriented_box_intersects_triangles,
)
from erc_phase1_solution.motion_profiles import IK_JOINTS, RIGHT_ARM_JOINTS, RIGHT_HOME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference-test', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('box_sat_reference', args.reference_test)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    root = args.source_root
    urdf = root / 'erc_description/urdf/tiago_pro.urdf'
    book_file = root / 'erc_description/models/book/sdf/erc_book.sdf'
    book_xml = ET.parse(book_file).getroot()
    size = np.fromstring(book_xml.find('.//collision/geometry/box/size').text, sep=' ')
    dimensions = size[[2, 1, 0]]
    chain = URDFChain.from_urdf(urdf, 'base_footprint', 'gripper_left_grasping_link', IK_JOINTS)
    right_chain = URDFChain.from_urdf(
        urdf, 'base_footprint', 'gripper_right_grasping_link',
        ('torso_lift_joint', *RIGHT_ARM_JOINTS))
    head_chain = URDFChain.from_urdf(
        urdf, 'base_footprint', 'head_front_camera_link',
        ('torso_lift_joint', 'head_1_joint', 'head_2_joint'))
    paths = {
        'omni_base_description': root / 'omni_base_robot/omni_base_description',
        'tiago_pro_description': root / 'tiago_pro_robot/tiago_pro_description',
        'tiago_pro_head_description': root / 'tiago_pro_head_robot/tiago_pro_head_description',
        'pal_sea_arm_description': root / 'pal_sea_arm/pal_sea_arm_description',
    }
    links = ('base_link', 'torso_base_link', 'torso_lift_link',
             *(f'arm_left_{i}_link' for i in range(1, 8)),
             *(f'arm_right_{i}_link' for i in range(1, 8)),
             'head_1_link', 'head_2_link', 'head_front_camera_link')
    meshes = load_urdf_collision_meshes(urdf, links, paths.__getitem__)
    candidate = json.loads(args.candidate.read_text())
    grasp = np.asarray(candidate['approach'][-1])
    grasp_tf = chain.forward(grasp)
    front = np.asarray(candidate['front'])
    world = front + [.5 * dimensions[0], 0., 0.] + reference.SIGNS * (.5 * dimensions + .015)
    attached = (world - grasp_tf[:3, 3]) @ grasp_tf[:3, :3]
    route = [*candidate['extraction'], *candidate['support_route'], *candidate['compact_route']]
    queries = []
    samples = 0
    facets_before = facets_after = 0
    for start, end in zip(route, route[1:]):
        # Every recorded section is sampled at the production density. This
        # benchmark does not alter the planner's sampling or decision order.
        for fraction in np.linspace(0., 1., 61):
            q = np.asarray(start) + (np.asarray(end) - start) * fraction
            samples += 1
            tf = chain.forward(q)
            corners = attached @ tf[:3, :3].T + tf[:3, 3]
            bounds = np.asarray([corners.min(axis=0), corners.max(axis=0)])
            transforms = chain.link_transforms(q)
            transforms.update(right_chain.link_transforms(np.r_[q[0], RIGHT_HOME]))
            transforms.update(head_chain.link_transforms([q[0], 0., -.1]))
            center, axes, half = oriented_box_from_corners(corners)
            for mesh in meshes:
                transform = transforms[mesh.link]
                local_center = np.mean(mesh.bounds, axis=0)
                local_half = .5 * (mesh.bounds[1] - mesh.bounds[0])
                world_center = transform[:3, :3] @ local_center + transform[:3, 3]
                world_half = np.abs(transform[:3, :3]) @ local_half
                if (np.any(world_center + world_half < bounds[0])
                        or np.any(world_center - world_half > bounds[1])):
                    continue
                surface = mesh.triangles @ transform[:3, :3].T + transform[:3, 3]
                vertices = (surface - center) @ axes
                kept = ~np.any((vertices.max(axis=1) < -half - 1e-9)
                               | (vertices.min(axis=1) > half + 1e-9), axis=1)
                facets_before += len(surface)
                facets_after += int(kept.sum())
                queries.append((corners, surface, mesh.watertight))
    if not queries:
        raise RuntimeError('No production broad-phase candidates generated')
    outcomes = {}
    timings = {'reference': [], 'filtered': []}
    functions = {'reference': reference.unfiltered_reference,
                 'filtered': oriented_box_intersects_triangles}
    for repeat in range(args.repeats):
        # Alternate measurement order to reduce systematic warmup bias.
        for name in (('reference', 'filtered') if repeat % 2 == 0
                     else ('filtered', 'reference')):
            began = time.perf_counter()
            result = [bool(functions[name](c, s, closed_surface=closed))
                      for c, s, closed in queries]
            timings[name].append(time.perf_counter() - began)
            if outcomes and result != next(iter(outcomes.values())):
                raise AssertionError('Collision verdict changed')
            outcomes[name] = result
    report = {
        'scope': 'Pure current-mesh kernel benchmark, historical representative joint route; no physical/path validation',
        'source_root': str(root), 'candidate': str(args.candidate),
        'urdf_sha256': hashlib.sha256(urdf.read_bytes()).hexdigest(),
        'book_sha256': hashlib.sha256(book_file.read_bytes()).hexdigest(),
        'book_dimensions_base_frame_m': dimensions.tolist(),
        'sampled_joint_states': samples, 'collision_queries': len(queries),
        'intersecting_queries': sum(outcomes['filtered']),
        'all_results_equal': outcomes['reference'] == outcomes['filtered'],
        'sat_facets_before': facets_before, 'sat_facets_after': facets_after,
        'timings_seconds': timings,
        'median_speedup': statistics.median(timings['reference']) / statistics.median(timings['filtered']),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
