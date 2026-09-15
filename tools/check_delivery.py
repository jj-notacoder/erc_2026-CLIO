#!/usr/bin/env python3
"""Read one simultaneous Gazebo pose frame after mission termination.

This evaluator subscribes only; it publishes no robot or simulator commands.
The result describes sampled containment, not long-term stability or scoring.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import queue
import time
import xml.etree.ElementTree as ET

import numpy as np


BOOK = None
BIN = 'erc_collection_bin'
TOPIC = '/world/erc_world/dynamic_pose/info'
HALF = np.array([.125, .01, .08])
LOWER = np.array([-.145, -.095, -.245])
UPPER = np.array([.145, .105, .245])
BIN_MESH_SHA256 = '7c036fe096a50eadc8c32f30837964c3bbbb012cf7a45b874eb7a6f96973aa2a'


def rotation(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1.) > 1e-5:
        raise ValueError('invalid unit quaternion')
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def evaluate(poses, stamp_ns):
    book, bin_pose = poses[BOOK], poses[BIN]
    bp, cp = (np.asarray(p['position'], dtype=float) for p in (book, bin_pose))
    if any(p.shape != (3,) or not np.isfinite(p).all() for p in (bp, cp)):
        raise ValueError('invalid model position')
    local = np.array(list(itertools.product((-1., 1.), repeat=3))) * HALF
    world = local @ rotation(book['orientation_xyzw']).T + bp
    corners = (world-cp) @ rotation(bin_pose['orientation_xyzw'])
    low, high = corners.min(axis=0), corners.max(axis=0)
    clearances = np.concatenate((low-LOWER, UPPER-high))
    return dict(
        pose_ros_ns=stamp_ns,
        simultaneous_model_poses=True,
        book_world=book, bin_world=bin_pose,
        book_half_extents_m=HALF.tolist(),
        conservative_bin_local_bounds_m=[LOWER.tolist(), UPPER.tolist()],
        book_corners_world_m=world.tolist(), book_corners_in_bin_m=corners.tolist(),
        book_bin_bounds_m=[low.tolist(), high.tolist()],
        face_order=['lower_x', 'lower_y_floor', 'lower_z', 'upper_x', 'upper_y_rim', 'upper_z'],
        signed_core_face_clearances_m=clearances.tolist(),
        minimum_clearance_m=float(clearances.min()),
        whole_book_inside_conservative_core=bool(np.all(clearances >= 0.)),
        footprint_inside_core=bool(np.all(clearances[[0, 2, 3, 5]] >= 0.)),
        floor_residual_m=float(clearances[1]),
        floor_support_consistent=bool(abs(clearances[1]) <= 1e-6),
        floor_support_analysis_tolerance_m=1e-6,
        scope='One simultaneous pose sample; no stability or competition score claim.',
    )


def verify_assets(root):
    files = dict(book_sdf=root/'book/sdf/erc_book.sdf',
                 bin_sdf=root/'collection_bin/sdf/erc_collection_bin.sdf',
                 bin_mesh=root/'collection_bin/meshes/erc_base_collection_bin.STL')
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    book = ET.parse(files['book_sdf']).getroot().find('model/link/collision')
    if book is None or not np.array_equal(np.fromstring(book.findtext('geometry/box/size'), sep=' '), HALF*2):
        raise ValueError('book collision dimensions do not match evaluator')
    if not np.array_equal(np.fromstring(book.findtext('pose', '0 0 0 0 0 0'), sep=' '), np.zeros(6)):
        raise ValueError('book collision has an unsupported model-frame offset')
    if hashes['bin_mesh'] != BIN_MESH_SHA256:
        raise ValueError('bin mesh differs from the run99 conservative-core reference')
    return dict(sha256=hashes, source_paths={k: str(v) for k, v in files.items()})


def snapshot(timeout):
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.transport13 import Node
    received = queue.Queue(maxsize=1)
    node = Node()

    def callback(message):
        if not received.empty():
            return
        selected = {name: [p for p in message.pose if p.name == name] for name in (BOOK, BIN)}
        if any(len(rows) != 1 for rows in selected.values()):
            return
        stamp = message.header.stamp.sec*1_000_000_000 + message.header.stamp.nsec
        if stamp <= 0:
            return
        poses = {}
        for name, rows in selected.items():
            p = rows[0]
            poses[name] = dict(entity_id=int(p.id),
                position=[p.position.x, p.position.y, p.position.z],
                orientation_xyzw=[p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
        try:
            received.put_nowait((poses, stamp))
        except queue.Full:
            pass

    node.subscribe(Pose_V, TOPIC, callback)
    try:
        return received.get(timeout=timeout)
    except queue.Empty as error:
        raise TimeoutError('no single Gazebo pose frame containing both named models') from error
    finally:
        node.unsubscribe(TOPIC)


def main():
    global BOOK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial-id', required=True)
    parser.add_argument('--target-model', required=True, help='Exact target_book_model from the trial summary')
    parser.add_argument('--timeout', type=float, default=20.)
    parser.add_argument('--asset-root', type=Path, default=Path('/opt/erc_ws/src/erc_description/models'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import re
    if re.fullmatch(r'book_col_[1-5]_row_[1-5]_(red|blue|green|yellow)', args.target_model) is None:
        parser.error('invalid target model name')
    BOOK = args.target_model
    if not 0. < args.timeout <= 60.:
        parser.error('--timeout must be in (0,60] wall seconds')
    assets = verify_assets(args.asset_root)
    started = time.monotonic()
    result = evaluate(*snapshot(args.timeout))
    result.update(trial_id=args.trial_id, target_model=BOOK, source_topic=TOPIC,
                  sampled_utc=datetime.now(timezone.utc).isoformat(),
                  sampling_wall_seconds=time.monotonic()-started, source_assets=assets)
    payload = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(payload)
    print(payload, end='')
    return 0 if result['whole_book_inside_conservative_core'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
