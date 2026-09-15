# Prepared lower-row lateral coverage

**Both scenarios passed complete offline PICK/carry checks on source `0a5fa41`.**
The row-3 case took 32.980 s and the bottom case took 57.841 s. Controller-call
lists are empty and both geometry owners closed. This establishes the tested
synthetic geometry cases; physical missions remain unverified.

These inputs are **synthetic onboard-aligned scenarios**, not measured marker
poses, detected book points, live odometry, or proof that navigation reaches them.
They isolate two tight lateral book/bay relationships from the seed 101 layout.

## Derivation

The read-only seed audit maps printed columns to physical columns as
`[3,5,2,1,4]` from physical left to right. Printed column 3 is physical column 1.
The launcher places each book at its physical column's marker Y plus its spawned
jitter. An ideal base frame with inward `+X` and shelf-left `+Y` therefore keeps
the exact offset by setting `marker_y = front_y - jitter`.

| Target | Saved front XYZ (m), realigned XY | Spawn lateral jitter (m) | Synthetic marker XYZ (m) |
|---|---|---|---|
| `3:yellow` (row 3) | `[.670,-.056,.9278333794123657]` | `-.142738515131016` | `[.605,.086738515131016,2.26]` |
| `3:green` (bottom) | `[.670,-.056,.604]` | `-.139151875236880` | `[.605,.083151875236880,2.26]` |

Front heights come from the audit's saved lower-row fixture heights. The nominal
spawn heights `.935` and `.605` are metadata, not substituted live observations.
Marker X `.605` follows the independent nominal lower fixture; marker Z `2.26`
is the ideal plate origin. Actual digit-centroid/TF errors are not sampled here.

Physical column 1's raw side walls are `[-.430,.570]` relative to the marker.
The unchanged `.200` lateral uncertainty and `.015` margin yield `[-.215,.355]`.
Subtracting the upright book's `.010` halfwidth gives initial book-only reserves
`.062261484868984` and `.06584812476312032` m. Those numbers exclude fingers,
arms, padding and motion. The complete planner must independently admit those.

The registered lip is marker + `.001016*inward`; the earlier of this and
`front-.065*inward` is the exclusion plane. Normal uncertainty `.010`, roof
uncertainty `.010`, bay margin `.015`, tool allowance `.005`, book padding `.015`,
carried shelf margin `.020`, official book dimensions `[.16,.02,.25]`, and the
existing sample grids remain unchanged. Floor height retains the existing
explicit upright supported-book assumption. The local bay does not certify
unobserved neighboring books or the entire shelf mesh.

## Harness scope

`guarded_coverage.py` imports only current production modules when `--run` is
explicitly selected. It uses `_official_manipulation_planner`, restores the
ordinary measured-joint seed, and wraps unchanged official robot facets in the
runtime immutable model owners. Head pitches are the ordinary row commands:
`-.40` for row 3 and `-.70` for bottom; the remaining arm/head context is the
existing HOME/right-HOME fixture with `.069` open aperture.

It invokes the full `plan_lower_shelf_pick` admission, including opening/torso,
registered empty setup, dense body/tool/bay entry, open gripper approach,
lift/withdrawal, deferred carry, complete cached tool route, extraction volume,
and the ordinary unloaded recovery proposal. It additionally checks supported
carry legs and the ordinary look-bin head sweep at the synthetic carried endpoint.
No predicate or solver is replaced. Controller methods fail immediately if called;
there is no ROS node initialization or live subscription.

The lift callback uses the ordinary `_lift_first_measurements` and
`run_pickup_geometry` continuation. The default uses the production eight-worker
empty and loaded owners, with the empty owner closed before the loaded owner.
`--serial` selects the existing serial geometry backend while keeping predicates.
Timestamps and stationary odometry are frozen logical fixture values, so this
does **not** establish real sensor freshness. A timer requests ordinary cooperative
cancellation after 300 wall seconds (configurable up to 600); normal worker cleanup
remains active. This timer is not a hard bound on an uninterruptible solver call.

Planning failure is an expected possible result. A PASS would establish only this
synthetic geometry case. It would not establish physical pickup, retained book
contact, placement, complete mission time, or safety of generic retained-failure
recovery. The returned unloaded recovery proposal is not a certificate allowing
blind loaded torso lowering.

## Replay invocation

Copy this directory to the container's `/tmp` first. Run **one target per process**
only after root releases the compute slot. Example shell body in `erc_friend_demo`:

```bash
source /opt/erc_ws/install/setup.bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ROS_DOMAIN_ID=140
python3 /tmp/erc_seed101_lower_lateral_coverage/guarded_coverage.py --run 3:yellow
```

Use `--run 3:green` in a separate process for the bottom scenario. `--describe`
only prints the scenario metadata and does not import ROS/numpy or run geometry.
The two result JSON files record the subsequently executed probes.

Each result records all current package source hashes, URDF and installed geometry
identity, exact synthetic context, stage/failure diagnostics, selected complete
plan and metrics, owner completion events, and controller-call count. Initial
preparation hashes are separate from run-time hashes because source may change
before authorized execution. Nonfinite diagnostic metrics are explicit strings
in otherwise strict JSON. Both executed cases completed against current production APIs.
