#!/usr/bin/env python3
"""Temporary offline validation for vertical top-row transport."""

import importlib.util

import numpy as np


path = '/opt/erc_ws/src/erc_phase1_solution/test/test_shutdown.py'
spec = importlib.util.spec_from_file_location('test_shutdown', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

np.set_printoptions(precision=12, suppress=True)
fronts = (
    np.asarray([0.6937779125, -0.05493238067, 1.58418506676]),
)
for front in fronts:
    node = module._official_manipulation_planner()
    node.carried_cradle_roll = -1.10
    node.carried_supported_jaw_vertical_component = 0.70
    node.carried_cradle_transfer = 0.75
    _, _, rotations, solutions, orientation_index, _, _ = (
        module._top_row_pick_plan(node, front)
    )
    result = node._plan_carried_return(
        front,
        solutions[-1],
        solutions[1],
        rotations[orientation_index],
        0.35,
    )
    lowering, cradle, roll_index, transport, corners = result
    cache = node._cached_post_retreat_plan
    previous = np.asarray(cache['start'], dtype=float)
    for waypoint, _ in cache['legs']:
        assert node._carried_post_retreat_transition_is_safe(
            previous,
            waypoint,
            corners,
            float(cache['shelf_front_x']),
        )
        previous = np.asarray(waypoint, dtype=float)
    print(
        'PASS', front.tolist(),
        'cradle', len(cradle),
        'roll_index', roll_index,
        'cached', len(cache['legs']),
        'extension_terminal', cache['legs'][1][0].tolist(),
        'radius', cache['compact_radius'],
        'jaw_vertical', node.chain.forward(cache['legs'][4][0])[2, 1],
    )
