# Bottom registered PLACE: complete offline proof

**The promoted bottom PLACE planner passed the full bounded offline search.**
It used 48 IK solves and two full candidates in 50.333 s. Physical bottom
retention, delivery and mission time remain unverified.

## Scope and inputs

This is a **synthetic combination of saved bottom carry geometry and recorded
blue placement observations**, not an observed bottom delivery. The bottom
carry/attachment came from a complete guarded offline PICK fixture. The bin,
table, stationary base, parked right arm/head and gripper aperture came from
blue trial `105d91426ddd`. The bottom identity, contact correlation, retained
flags and logical freshness stamps are explicitly synthetic fixture state.
All ordinary planning admission checks execute, but those assumed states do
not prove live contact, freshness or retention.

- Bottom target identity: `book_col_1_row_5_green`; no target-dependent pose or
  saved Cartesian solution is injected into the solver.
- Original measured-front fixture: `[0.67, -0.056, 0.604]` m.
- Official book dimensions: `[0.16, 0.02, 0.25]` m; collision padding 0.015 m.
  Recomputed attached corners agree exactly with the saved fixture.
- Exact original negative-wrist carry and staging states are in
  [bottom_carry.json](bottom_carry.json); staging is only an IK seed.
- Recorded scene and master aperture are preserved in
  [recorded_place_inputs.json](recorded_place_inputs.json).

## Result

| Check | Result |
|---|---|
| Full planner | PASS, 50.333 s planning; 50.538 s total |
| Search | 48 IK calls; 2 full candidates; 1 full-candidate rejection |
| Limits | Unchanged 512 total IK, 192 shared anchor allowance, 6 full candidates |
| Setup | All 22 controller-sized loaded legs admitted |
| Cartesian path | All 21 poses and connecting loaded transitions admitted |
| Opening | Full 52-sample loaded opening sweep and static unloaded endpoint admitted |
| Release identity | Exact current Request and selected endpoint verified |
| Scene checks | 3,890 samples, 177 exact cache hits |
| Selected target | 25 mm near-side shift; exact selected pose propagated |
| Worst padded-book wall reserve | Unchanged at 0.03583510190354866 m |
| Support and wrist | Original 0.75 dense threshold; all setup/Cartesian wrists negative |
| Processes | Zero controller calls; zero geometry children; exit 0; cleanup scan empty |

The solver separately screened 52 opening samples for each of its two
candidates (104 samples in its search counters). That is distinct from the
full candidate's opening sweep. The complete original output, inputs, route,
stage events and source hashes are in
[bottom_place_result.json](bottom_place_result.json). The result retains its
original `current_production_proof: false` label: it was executed as a private
draft, whose exact three runtime files were subsequently promoted unchanged.
Production integration checks are recorded separately by the parent task.

## Change and retained checks

Extra proposals require the exact existing release-only Request, its currently
held canonical stock row-5 target and strict verified transport, support and
payload-monitor flags. Current contact/target/held-object/scene checks remain
owned by that Request. There is no new cross-phase controller owner, stored
pose override, column/color gate or weakened retention check.

The original negative-wrist solver may now use the existing registered
center-first target proposals and high-point anchor search for this eligible
bottom request. Budgets stay shared. After the original four setup policies
fail, a fifth bounded proposal selects the nearest legal negative wrist angle
that meets signed support plus a 0.005 waypoint reserve. It preserves the
current start, proximal route and exact final IK endpoint. Original dense
support, body, table, bin, opening and static release checks still decide
acceptance. Other rows retain their existing proposal scope.

[setup_diagnosis.json](setup_diagnosis.json) contains compact excerpts from the
earlier saved-path diagnosis: the original maximum-support wrist collided with
`head_2_link`; nearest support without the reserve dipped below 0.75 during
interpolation and was rejected; adding the proposal reserve allowed a fully
checked route. These diagnostics are supporting evidence only. The full proof
above regenerated its path through the actual bounded IK solver.

## Software evidence and files

- [portable_tests.log](portable_tests.log): 128 tests passed in 0.17 s before
  promotion, covering all 20 canonical bottom identities, all 60 other-row
  identities, actual Request invalidation, proposal math/cancellation, original
  full admission and selected-target propagation. Four actual solver tests
  cover negative IK filtering, center-first selection, shared small budgets
  and clearing the registered index before ordinary fallback. Test geometry is
  synthetic. Test modules are in the repository's normal `test` directory.
- [promoted_manifest.json](promoted_manifest.json): exact promoted runtime and
  test hashes, plus the historical source-inverse fixture and pinned helper.
- [runtime.patch](runtime.patch): the isolated three-file runtime change.
- `runtime_snapshot/`: exact three planner files used by this run. These small
  snapshots allow replay after production source moves forward.
- [manifest.json](manifest.json): hashes of every packaged artifact.
- [cleanup.json](cleanup.json): completed process cleanup.

## Replay without changing the original harness

[replay.py](replay.py) is preserved byte-for-byte, including its recorded
SHA-256 `76bd85e33dcd1adf5a01232086babb3ab8202d540faa63fc8155cd6982e69696`.
It already resolves its three draft modules and two inputs relative to its own
file; only the original temporary directory layout needs restoring. Copy this
evidence directory into the configured ROS container, then copy its snapshot
modules alongside the harness:

```sh
docker cp docs/evidence/local_gazebo_20260915/bottom_registered_place erc_friend_demo:/tmp/
docker exec erc_friend_demo bash -c 'cp /tmp/bottom_registered_place/runtime_snapshot/*.py /tmp/bottom_registered_place/'
docker exec -e PYTHONDONTWRITEBYTECODE=1 -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e ROS_DOMAIN_ID=140 erc_friend_demo /entrypoint.sh bash -c 'source /opt/erc_ws/install/setup.bash; python3 /tmp/bottom_registered_place/replay.py bottom --workspace /opt/erc_ws --cap-seconds 180'
```

Run only while no simulation or other heavy planner is active. The script
writes a new result into its temporary copy. To reproduce the exact recorded
case, keep the official model/profile and remaining solution dependencies at
the hashes in the original result. The three snapshot modules override their
old `source_sha256` entries and instead use `draft_sources`. No Gazebo entity
pose or controller call is needed. Later dependency changes constitute a new
validation run and must not be described as an identical replay.
