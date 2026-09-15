# Registered placement from a positive-wrist carry

The integrated official-model replay passes all planning checks. It selects a target 50 mm toward the near side of the measured bin, while preserving the original minimum wall reserve. **This is an offline planning result, not a live mission completion.**

## Result

| Check | Result |
|---|---:|
| IK calls / unchanged limit | 29 / 512 |
| Complete candidates | 1 |
| Setup legs / Cartesian waypoints | 21 / 19 |
| Exact scene samples | 2,583 |
| Nominal / selected minimum padded-book wall reserve | 35.833 / 35.833 mm |
| Effective bin material and registration margin | 14.167 mm |
| Computed maximum shift preserving that reserve | 55 mm |
| Selected shift, after center and 25 mm proposals failed | 50 mm |
| Controller calls | 0 |

The full carried setup, every loaded segment, signed support, robot/tool/book/bin/table geometry, opening sequence, static open-hand endpoint and release identity checks passed. The final setup and Cartesian joint arrays exactly match the earlier complete geometry proposal.

The promoted source also passed **353 tests across 14 relevant modules in 7.46 s**. See [the test log](promoted_source_checks.log). Those tests cover target geometry, positive-policy and negative-default solver behavior, candidate/fallback selection, budgets, cancellation, caller propagation, scene planning, backend handling and release-only completion.

## Change

The old solver required a negative wrist, while row 3 deliberately carries with a positive wrist. Removing that filter alone was insufficient: the original centered high corner failed the bounded reach probes, and a reachable near-side target still exceeded the unchanged 0.40-radian step limit on its last coarse approach segment.

For the admitted measured-positive family, the planner now tries the center followed by increasing 25 mm offsets along the registered bin's long axis toward the robot. The maximum shift is computed from all eight padded book corners and the exact existing bin margins, without reducing the nominal worst horizontal wall reserve. Orientation and the configured release/clearance heights stay unchanged. Each proposal retains its original Cartesian segment endpoints and adds a midpoint.

Tight IK starts at the high, far corner and continues backward and forward into execution order. Proposals share the existing 512-call budget; the anchor phase reserves at most 192 calls. The original 0.40-radian step, 2 mm position residual, 0.02-radian orientation residual, 0.10-radian joint margin, 0.75 signed support threshold and complete geometry gates remain mandatory. Default negative-start/top-row behavior and its centered target are unchanged.

The returned target propagates to PLACE's release and center fields. Execution and release proof use the admitted joint solutions.

## Files

- [Validated summary](validated_summary.json): compact metrics, limitations and hashes.
- [Final integrated result](integrated_positive_115500.json): inputs, solved routes, exact scene diagnostics and IK trace.
- [Event log](integrated_positive_115500.jsonl): planner milestones and target selection.
- [Source manifest](source_manifest.json): the four changed runtime files and their prior/source hashes.
- [Portable replay](replay.py): uses the checked-in raw evidence in [column4_blue_transport](../column4_blue_transport/), without a simulator, private `/tmp` files or ignored live journals.

## Limits of this evidence

The 115.500 s camera scene is inferred from independent recorder ordering; the historical trial did not log the exact selected PLACE epoch. Its final measured carry joints and PLACE attachment/context were also missing. This replay uses the recorded pick target and grasp IK to reconstruct the attachment, production code to regenerate staging, the final commanded positive carry, commanded parked head/right arm, and recorded retained gripper aperture. It uses the selected profile's actual **0.35 m torso height**. Earlier 0.30 m probes were counterfactual diagnostics and are not included here.

This replay runs the actual official geometry serially. Live planning uses the managed worker backend. Its external 180-second diagnostic alarm is distinct from the unchanged runtime 1,800-second wall budget and 512-call IK cap. The recorded 32.15-second serial planning duration is not a live mission timing guarantee. No simulator ground-truth pose is used as a planner input.

## Reproduce

Use the project's prepared container and installed ROS environment. The script checks the four pinned runtime hashes before initializing the model. Run from the workspace root, with no simulation or other heavy planner active:

```sh
export ROS_DOMAIN_ID=137
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
source install/setup.bash
python3 docs/evidence/local_gazebo_20260915/positive_registered_place/replay.py \
  --output-dir /tmp/erc_positive_place_replay --trace
```

The archived replay was run before portability-only edits to paths and output arguments. The portable script was syntax-checked; it was not rerun to avoid duplicating the completed geometry work before the next live trial.
