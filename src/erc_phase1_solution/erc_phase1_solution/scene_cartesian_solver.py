"""Bounded signed-support Cartesian search; no ROS, commands, or saved route.

Scene checks here screen endpoints and nominal opening. The caller's mandatory
candidate validator must check actual setup, every intervening loaded segment,
and unloaded return before a result can be accepted. A finite beam is a search
policy, not a proof that rejected targets are unreachable.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import time

import numpy as np


@dataclass(frozen=True)
class SearchLimits:
    max_ik_calls: int = 512
    max_paths: int = 24
    max_candidates: int = 6
    wall_seconds: float = 420.0


class SceneCartesianSearchError(RuntimeError):
    def __init__(self, reason, diagnostics):
        super().__init__(reason)
        self.diagnostics = diagnostics


def _array(value, shape, name):
    result = np.asarray(value, dtype=float).copy()
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f'invalid {name}')
    return result


def solve_scene_cartesian(node, positions, rotation, torso_height, staging_seed,
                          scene, candidate_validator, *, measured_seed, aperture,
                          open_aperture, seed_templates=(), limits=SearchLimits(),
                          diagnostics_out=None, wrist_policy="negative",
                          position_proposals=()):
    """Solve supplied base-frame TCP poses and validate each complete candidate.

    Inputs are caller-snapshotted measured/staging/template seeds, not commands
    and not a recorded successful path. `candidate_validator(path, [])` follows
    the existing `_solve_cartesian_path` callback contract. False tries another
    retained path; exceptions propagate. Returns the old four-tuple
    `(solutions, orientation_index, path_score, [])`. Optional diagnostics are
    copied into `diagnostics_out`; failures carry diagnostics on the exception.
    Cancellation is checked before/after
    every expensive callback and immediately before successful return.
    """
    if wrist_policy not in ('negative', 'measured_positive'):
        raise ValueError('invalid Cartesian wrist policy')
    if not callable(candidate_validator):
        raise ValueError('full candidate validator is required')
    if diagnostics_out is not None and not isinstance(diagnostics_out, dict):
        raise ValueError('diagnostics_out must be a dictionary')
    if (not isinstance(limits, SearchLimits)
            or not 1 <= limits.max_ik_calls <= 4096
            or not 1 <= limits.max_paths <= 64
            or not 1 <= limits.max_candidates <= 16
            or not math.isfinite(limits.wall_seconds)
            or not 0 < limits.wall_seconds <= 1800):
        raise ValueError('invalid bounded search limits')
    if any(isinstance(v, bool) or not isinstance(v, int) for v in
           (limits.max_ik_calls, limits.max_paths, limits.max_candidates)):
        raise ValueError('search counts must be integers')
    raw_positions = np.asarray(positions, dtype=float)
    if (raw_positions.ndim != 2 or raw_positions.shape[1:] != (3,)
            or not 1 <= len(raw_positions) <= 64 or not np.all(np.isfinite(raw_positions))):
        raise ValueError('invalid Cartesian positions')
    points = raw_positions.copy()
    orientation = _array(rotation, (3, 3), 'rotation')
    if (not np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-8, rtol=0)
            or abs(float(np.linalg.det(orientation))-1.) > 1e-8):
        raise ValueError('rotation is not proper orthonormal')
    torso = float(torso_height)
    aperture, open_aperture = float(aperture), float(open_aperture)
    if not all(math.isfinite(v) for v in (torso, aperture, open_aperture)):
        raise ValueError('nonfinite torso/aperture')
    # Accepted raw stock feedback may have its existing 1 um endpoint allowance.
    # Public opening remains in the exact official command range.
    if not -1e-6 <= aperture <= .069+1e-6 or not 0 <= open_aperture <= .069:
        raise ValueError('invalid measured or commanded aperture')
    chain = node.chain
    required_names = ['torso_lift_joint'] + [f'arm_left_{i}_joint' for i in range(1, 8)]
    if list(chain.active_names) != required_names:
        raise ValueError('unexpected arm chain ordering')
    lower, upper = _array(chain.lower, (8,), 'lower bounds'), _array(chain.upper, (8,), 'upper bounds')
    configured_margin = float(node.place_joint_limit_margin)
    configured_support = float(node.carried_supported_jaw_vertical_component)
    if not math.isfinite(configured_margin) or not math.isfinite(configured_support):
        raise ValueError('nonfinite configured margin/support')
    margin = max(.10, configured_margin)
    support = max(.75, configured_support)
    if (support > 1.
            or not lower[0] <= torso <= upper[0]
            or np.any(lower[1:]+margin >= upper[1:]-margin)):
        raise ValueError('invalid arm bounds/support policy')
    if (wrist_policy == 'measured_positive'
            and _array(measured_seed, (8,), 'measured IK seed')[-1] <= 0.):
        raise ValueError('positive-start wrist policy requires a positive measured wrist')
    seeds = []
    for value in (measured_seed, staging_seed, *seed_templates):
        seed = _array(value, (8,), 'IK seed')
        seed[1:] = np.clip(seed[1:], lower[1:]+margin, upper[1:]-margin)
        seed[0] = torso
        if not any(np.array_equal(seed, old) for old in seeds):
            seeds.append(seed)
    if len(seeds) > 8:
        raise ValueError('at most eight distinct seeds are supported')
    matrices = []
    for point in points:
        matrix = np.eye(4); matrix[:3, :3] = orientation; matrix[:3, 3] = point
        matrices.append(matrix)
    if not isinstance(position_proposals, (tuple, list)) or len(position_proposals) > 12:
        raise ValueError('invalid bounded Cartesian target proposals')
    if position_proposals and wrist_policy != 'measured_positive':
        raise ValueError('target proposals require admitted positive measured carry')
    proposal_matrices = []
    for proposed in position_proposals:
        values = np.asarray(proposed, dtype=float)
        if (values.ndim != 2 or values.shape[1:] != (3,) or not 3 <= len(values) <= 64
                or not np.all(np.isfinite(values))):
            raise ValueError('invalid Cartesian target proposal')
        owned = []
        for point in values:
            matrix = np.eye(4); matrix[:3, :3] = orientation; matrix[:3, 3] = point
            owned.append(matrix)
        proposal_matrices.append(owned)
    started = time.monotonic()
    calls = candidates = endpoint_samples = opening_samples = 0
    reasons = Counter()
    levels = []
    checked_candidates = set()
    strategy = dict(loose_paths=0, refined_paths=0)
    completion = None
    completion_active = False
    completion_start = 0

    def diagnostics():
        detail = dict(search='bounded_negative_wrist_variable_swivel_v1', ik_calls=calls,
                    candidates=candidates, endpoint_samples=endpoint_samples,
                    opening_samples=opening_samples, rejections=dict(reasons), levels=levels,
                    wall_seconds=time.monotonic()-started, wall_budget_seconds=limits.wall_seconds,
                    position_tolerance=.002,
                    orientation_tolerance=.02, maximum_joint_step=.40,
                    joint_margin=margin, minimum_signed_support=support,
                    full_candidate_validator_required=True, recorded_joint_path_used=False,
                    strategy=dict(strategy))
        if wrist_policy == 'measured_positive':
            detail['search'] = 'bounded_signed_support_positive_start_v1'
            detail['wrist_policy'] = wrist_policy
        if completion is not None:
            detail['completion_first'] = dict(completion)
            if completion_active:
                detail['completion_first']['ik_calls'] = calls-completion_start
        return detail

    def checkpoint():
        if node._cancel.is_set():
            raise SceneCartesianSearchError('scene_cartesian_cancelled', diagnostics())
        if time.monotonic()-started >= limits.wall_seconds:
            raise SceneCartesianSearchError('scene_cartesian_wall_budget', diagnostics())

    def solve(matrix, seed, elbow=None, *, loose=False):
        nonlocal calls
        checkpoint()
        if calls >= limits.max_ik_calls:
            raise SceneCartesianSearchError('scene_cartesian_ik_budget', diagnostics())
        fixed = {'torso_lift_joint': torso}
        if elbow is not None:fixed['arm_left_3_joint'] = elbow
        calls += 1
        q, score = chain.solve(matrix, [seed.copy()], position_tolerance=.012 if loose else .002,
                               orientation_tolerance=.10 if loose else .02, max_iterations=220,
                               fixed_positions=fixed, joint_limit_margin=margin)
        checkpoint()
        if q is None:
            reasons['ik'] += 1
            return None
        q = _array(q, (8,), 'IK output')
        if not math.isfinite(float(score)):
            raise ValueError('nonfinite IK score')
        if (np.any(q < lower) or np.any(q > upper)
                or np.any(q[1:] < lower[1:]+margin) or np.any(q[1:] > upper[1:]-margin)
                or abs(q[0]-torso) > 1e-9
                or (elbow is not None and abs(q[3]-elbow) > 1e-9)):
            reasons['limits_or_fixed_joint'] += 1
            return None
        return q

    def endpoint(q, matrix):
        nonlocal endpoint_samples
        checkpoint()
        if wrist_policy == 'negative' and q[-1] >= 0:
            reasons['nonnegative_wrist'] += 1
            return False
        actual = _array(chain.forward(q), (4, 4), 'forward pose')
        error = np.asarray(chain.pose_error(actual, matrix), dtype=float)
        if error.shape != (6,) or not np.all(np.isfinite(error)):
            raise ValueError('invalid forward residual')
        if np.linalg.norm(error[:3]) > .002+1e-10 or np.linalg.norm(error[3:]) > .02+1e-10:
            reasons['pose_error'] += 1
            return False
        if actual[2, 1] < support:
            reasons['signed_support'] += 1
            return False
        endpoint_samples += 1
        passed = scene.sample(q.copy(), aperture, True)
        checkpoint()
        if not passed:reasons['endpoint_scene'] += 1
        return bool(passed)

    def retain(frontier, path):
        q = path[-1]
        if any(np.max(abs(q-old[-1])) < .025 for old in frontier):return
        if sum(abs(old[-1][3]-q[3]) < 1e-6 for old in frontier) < 2:
            frontier.append(path)

    def admit(path):
        nonlocal candidates, opening_samples
        checkpoint()
        signature = tuple(q.tobytes() for q in path)
        if signature in checked_candidates:return None
        checked_candidates.add(signature)
        count = int(math.ceil(abs(open_aperture-aperture)/.001))+1
        for opening in np.linspace(aperture, open_aperture, count):
            checkpoint()
            opening_samples += 1
            passed = scene.sample(path[-1].copy(), float(opening), True)
            checkpoint()
            if not passed:
                reasons['opening_scene'] += 1
                return None
        if candidates >= limits.max_candidates:
            raise SceneCartesianSearchError('scene_cartesian_candidate_budget', diagnostics())
        candidates += 1
        # The validator gets owned copies, so it cannot mutate the selected
        # solution while checking its full execution route.
        passed = candidate_validator([q.copy() for q in path], [])
        checkpoint()
        if passed:
            score = sum(float(np.linalg.norm(b[1:]-a[1:])) for a, b in zip(path, path[1:]))
            detail = diagnostics(); detail['strategy'] = dict(strategy)
            if diagnostics_out is not None:diagnostics_out.update(detail)
            return [q.copy() for q in path], 0, score, []
        reasons['full_candidate'] += 1
        return None

    checkpoint()
    if wrist_policy == 'measured_positive' and len(matrices) >= 3:
        # Try the most extended high corner before committing to an easier
        # clearance branch. All proposals share the original global budgets.
        # Caller-proposed near-side targets retain the registered cavity margin.
        anchor_end = calls + min(192, max(0, (limits.max_ik_calls-calls)//2))
        strategy.update(anchor_roots=0, anchor_complete_paths=0,
                        position_proposal_index=None, anchor_ik_budget=anchor_end-calls)
        for proposal_index, proposed in enumerate(proposal_matrices or [matrices]):
            anchor_index = int(np.argmax([np.linalg.norm(matrix[:3, 3]) for matrix in proposed]))
            strategy['anchor_index'] = anchor_index
            for seed in seeds:
                checkpoint()
                if calls >= anchor_end:
                    break
                root = solve(proposed[anchor_index], seed)
                if root is None or not endpoint(root, proposed[anchor_index]):
                    continue
                strategy['anchor_roots'] += 1
                path = [None] * len(proposed)
                path[anchor_index] = root
                for indices in (range(anchor_index-1, -1, -1),
                                range(anchor_index+1, len(proposed))):
                    previous = root
                    for index in indices:
                        if calls >= anchor_end:
                            break
                        q = solve(proposed[index], previous)
                        if q is None:
                            break
                        if np.max(abs(q[1:]-previous[1:])) > .40:
                            reasons['anchor_joint_step'] += 1
                            break
                        if not endpoint(q, proposed[index]):
                            break
                        path[index] = q
                        previous = q
                    else:
                        continue
                    break
                if any(q is None for q in path):
                    continue
                strategy['anchor_complete_paths'] += 1
                strategy['position_proposal_index'] = proposal_index if proposal_matrices else None
                accepted = admit(path)
                if accepted is not None:
                    return accepted
            if calls >= anchor_end:
                break
        # The original fallback below uses only the original supplied target.
        strategy['position_proposal_index'] = None
    # A loose solve may discover a useful redundant branch which a direct
    # tight solve misses. These poses are seeds only: every waypoint is solved
    # again tightly, then checked with the tight FK residual and full scene.
    # The loose sequence itself can never reach the candidate validator.
    for seed in seeds:
        loose_path = []
        previous = seed
        for matrix in matrices:
            q = solve(matrix, previous, loose=True)
            if q is None or (wrist_policy == 'negative' and q[-1] >= 0):break
            if loose_path and np.max(abs(q[1:]-previous[1:])) > .40:break
            loose_path.append(q); previous = q
        if len(loose_path) != len(matrices):continue
        strategy['loose_paths'] += 1
        refined = []
        for matrix, loose_seed in zip(matrices, loose_path):
            q = solve(matrix, loose_seed)
            if q is None:break
            if refined and np.max(abs(q[1:]-refined[-1][1:])) > .40:
                reasons['refinement_joint_step'] += 1
                break
            if not endpoint(q, matrix):break
            refined.append(q)
        if len(refined) != len(matrices):continue
        strategy['refined_paths'] += 1
        accepted = admit(refined)
        if accepted is not None:return accepted

    # Unfixed solves discover current-target branches. Refine them and the
    # supplied seeds with a deterministic swivel grid; no stored q solution.
    refinements = []
    frontier = []
    for seed in seeds:
        q = solve(matrices[0], seed)
        if q is not None:
            refinements.append(q)
            if endpoint(q, matrices[0]):retain(frontier, [q])
    refinements.extend(seeds)
    for elbow in (-1., -1.125, -.875, -1.25, -.75, -1.5, -.5, 0., .5, 1., 1.5):
        if not lower[3]+margin <= elbow <= upper[3]-margin:continue
        for seed in refinements:
            q = solve(matrices[0], seed, elbow)
            if q is not None and endpoint(q, matrices[0]):retain(frontier, [q])
        if len(frontier) >= limits.max_paths:break
    frontier = frontier[:limits.max_paths]
    levels.append(dict(waypoint=0, surviving=len(frontier)))
    # Try completing existing tight roots before spending the remaining budget
    # on every swivel at every level. These are live target solves, never saved
    # q paths. Reserve at least half the remaining global calls for the original
    # beam fallback; all solves use its original clock and global count.
    completion_start = calls
    completion_cap = min(192, max(0, (limits.max_ik_calls-calls)//2))
    completion_end = calls+completion_cap
    completion = dict(policy='bounded_greedy_existing_delta_order_v1',
                      ik_budget=completion_cap, ik_calls=0, roots_attempted=0,
                      completed_paths=0, accepted=False)
    completion_active = True
    for root in frontier:
        checkpoint()
        if calls >= completion_end:
            break
        completion['roots_attempted'] += 1
        path = [q.copy() for q in root]
        while len(path) < len(matrices):
            previous = path[-1]
            chosen = None
            for delta in (0., -.125, .125, -.25, .25, -.375, .375):
                checkpoint()
                if calls >= completion_end:
                    break
                elbow = round(float(previous[3])+delta, 6)
                if not lower[3]+margin <= elbow <= upper[3]-margin:
                    continue
                q = solve(matrices[len(path)], previous, elbow)
                if q is None:
                    continue
                if np.max(abs(q[1:]-previous[1:])) > .40:
                    reasons['joint_step'] += 1
                    continue
                if endpoint(q, matrices[len(path)]):
                    chosen = q
                    break
            if chosen is None:
                break
            path.append(chosen)
        # The final solve may use exactly the local call allowance. It has
        # already passed tight FK/scene checks: still run the unchanged opening
        # and complete candidate gates before deciding whether to return it.
        if len(path) == len(matrices):
            completion['completed_paths'] += 1
            accepted = admit(path)
            if accepted is not None:
                completion['accepted'] = True
                completion['ik_calls'] = calls-completion_start
                completion_active = False
                if diagnostics_out is not None:
                    diagnostics_out['completion_first'] = dict(completion)
                return accepted
    completion['ik_calls'] = calls-completion_start
    completion_active = False
    for index, matrix in enumerate(matrices[1:], start=1):
        following = []
        for path in frontier:
            previous = path[-1]
            for delta in (0., -.125, .125, -.25, .25, -.375, .375):
                elbow = round(float(previous[3])+delta, 6)
                if not lower[3]+margin <= elbow <= upper[3]-margin:continue
                q = solve(matrix, previous, elbow)
                if q is None:continue
                if np.max(abs(q[1:]-previous[1:])) > .40:
                    reasons['joint_step'] += 1
                    continue
                if endpoint(q, matrix):retain(following, [*path, q])
        following.sort(key=lambda path:float(path[-1][3]))
        if len(following) > limits.max_paths:
            indices = np.linspace(0, len(following)-1, limits.max_paths).round().astype(int)
            following = [following[int(i)] for i in indices]
        frontier = following
        levels.append(dict(waypoint=index, surviving=len(frontier)))
        checkpoint()
        if not frontier:break
    else:
        for path in frontier:
            accepted = admit(path)
            if accepted is not None:return accepted
    checkpoint()
    raise SceneCartesianSearchError('scene_cartesian_no_accepted_route', diagnostics())
