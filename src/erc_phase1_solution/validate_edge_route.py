#!/usr/bin/env python3
"""Temporary corrected-frame edge-route arm/book validator."""

import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location(
    'test_shutdown_edge_validate',
    '/opt/erc_ws/src/erc_phase1_solution/test/test_shutdown.py',
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

front = np.asarray([.6927616089375521, -.055868643690940314, 1.5836944979660614])
node = helpers._official_manipulation_planner()
solutions = helpers._top_row_pick_plan(node, front)[3]
q5 = np.asarray(solutions[5], dtype=float)
qout = np.asarray([.35,.6734756853,.8960508912,-.0646125748,-1.3764016728,.2838232384,1.2350771320,-.5457885161])
qdown = np.asarray([.35,.7545204806,.6694487140,-.0416973493,-1.6824786171,.3423814816,1.3375676675,-.6026645814])
qroll = qdown.copy(); qroll[-1] -= 1.55
physical = node._attached_book_corners(front, solutions[-1])
q5_pose = node.chain.forward(q5)
stationary = physical @ q5_pose[:3, :3].T + q5_pose[:3, 3]
print('q5', q5.tolist(), 'book', np.min(stationary,axis=0).tolist(), np.max(stationary,axis=0).tolist())
for name, first, last in (
    ('out', q5, qout),
    ('down', qout, qdown),
    ('roll', qdown, qroll),
):
    for index, fraction in enumerate(np.linspace(0.0, 1.0, 101)):
        q = first + fraction * (last - first)
        self_hit = node._robot_self_collision(q)
        book_hit = node._carried_robot_collision(q, stationary)
        if self_hit is not None or book_hit is not None:
            raise RuntimeError(
                f'{name} failed at {index}: self={self_hit}, book={book_hit}'
            )
    print(name, 'PASS')
