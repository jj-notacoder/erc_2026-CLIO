# All-books development and validation

The original column 2/red implementation is preserved at tag `baseline-5m18`
(`69d6a9270b7f800a154e521a047d6671acde6fc4`). Work for all four shelf rows is on
`fix/all-books`. A successful geometry check is not a completed physical mission.

## Physical trials

All trials below use the unchanged official simulator, seed 101, and a visible
Gazebo viewer. Columns identify the requested printed marker.

| Trial | Requested book | Result | Solution launch to terminal | Including simulator startup |
|---|---|---|---:|---:|
| `9c768db2db9e` | Column 2, red | DONE; whole book inside bin, zero reported collisions | 318.006 s | 328.548 s |
| `d0fcc30b05eb` | Column 4, blue | Aborted during close-view depth reacquisition | 133.969 s | 144.444 s |
| `146e2f8aaa7b` | Column 4, blue | Reacquisition passed; aborted before grasp because the old lower-row lift route was infeasible | 126.496 s | 137.559 s |
| `180fb89230d6` | Column 4, blue | Plan passed; shelf contact during empty setup, unverified grasp, aborted after recovery | 317.413 s | 327.678 s |
| `121f7ff7e8ad` | Column 4, blue | Correct book grasped with bilateral contact; zero collisions; stopped before lift because a top-row timing option rejected the shorter route | 346.981 s | 357.056 s |

Trial `146e2f8aaa7b` used source `94dc436`. Close-view reacquisition took 0.7
simulated seconds. It then rejected the old lift route at leg 4. No grasp was
dispatched and zero collision episodes were reported. Source stayed unchanged
during the run and all owned simulation processes stopped afterward.

Trial `180fb89230d6` used source `9690fa8`. Its complete nominal lower-row
approach/lift/carry checks passed, but the empty setup lacked a full shelf
clearance check. Contacts identified arm link 6 against the shelf during entry
and arm link 5 during recovery. Closure stopped for unilateral contact without
acquiring the target. Two collision episodes were recorded. This route is not
validated for physical use; shelf-aware empty entry is being corrected. All
owned processes stopped normally, with source unchanged throughout the trial.
The [portable trial record](evidence/local_gazebo_20260915/column4_blue_lower/README.md)
includes its event log and pickup view.

Trial `121f7ff7e8ad` used source `1f5d677`. The corrected empty setup and
Cartesian approach completed without reported collision. The exact requested
book was latched, both fingers confirmed contact, and the measured initial-lift
geometry and fresh retention check passed. No lift was dispatched: the selected
3× withdrawal speed option required a five-leg top-row prefix, while this lower
route had three legs. The subsequent fix scopes that option to its admitted
top-row route; lower rows retain their original segment durations and gates.
The [grasp evidence](evidence/local_gazebo_20260915/column4_blue_grasp/README.md)
records the unchanged source, normal cleanup and independent pose observations.
This run included an extra read-only pose observer and does not establish
uninstrumented performance or successful delivery. The table uses solution
process startup; the solution's later reported launch origin gives 346.669 s.

## Current implementation

- Books-mode observations retain the selected numbered bay, confirmed row,
  observation timestamp, and fresh command epoch. Close-view detections must
  agree with that context.
- Depth fitting can explain a connected front face plus attached orthogonal
  side/top faces while retaining fit uncertainty and geometric consistency
  requirements. The captured failed blue view and synthetic four-colour,
  four-row views pass; corrupted geometry remains rejected.
- All three lower rows (rows 2–4) now have complete guarded offline plans.
  Each candidate must pass empty approach, lift, carried payload, supported
  compaction, detailed gripper motion and recovery checks before motion begins.
- Empty entry uses dense robot and gripper mesh checks against the registered
  shelf front, floor, roof, sides and rear panel. The front plane comes from
  the observed marker and retains the earlier book-derived bound, with a
  10 mm normal uncertainty allowance. Coarse setup samples only select
  proposals; every returned setup leg receives the full dense checks.
- Bottom-row entry rebuilds the approach while preserving the separately
  proposed grasp and short withdrawal joints. Lift planning uses that explicit
  withdrawal to retain the validated arm configuration.
- A lower-row failure that retains the book stops with the gripper closed when
  no complete loaded recovery has been admitted. An offline unloaded recovery
  route does not authorize recovery from an arbitrary loaded failure state.
- Exact immutable robot geometry is shared between collision checks and reused
  when mesh identity and transform bytes match. Each sample still captures its
  current head/right-arm context; collision checks and margins remain active.

### Software and offline evidence

Recorded focused checks include **102 unit tests for shared shelf checks and
bottom pickup**, plus **23 lower-shelf protocol/runtime tests**. A later focused run passed all
**14 recovery and runtime wiring tests**, including two earlier runtime cases.
The three complete official-model lower-row cases also pass against current
source; row two was rerun after correcting its old expected carry proposal.
These cover required dense admission,
registration and uncertainty bounds, cancellation, preserved withdrawal joints,
and stopping when loaded recovery is unverified. These counts cover the focused
checks listed here.

| Offline validation job | Wall time | Result |
|---|---:|---|
| Current row-2 complete harness | 56.12 s | Passed |
| Row-3 integrated complete harness | 53.32 s | Passed |
| Bottom complete test run | 63.89 s | 26 tests passed, including the complete official-model case |

The [offline evidence](evidence/lower_shelf_offline_20260915/README.md) preserves
the input states, route results and source identities. These durations measure
local offline validation work. Physical mission times
and outcomes remain in the trial table above. The lower-row checks include
complete lift/carry planning, registered entry, signed finger support, compact
navigation radius, unloaded recovery and the look-bin head sweep. They do not
establish physical grasp retention, placement, or delivery for these rows.

Delivery for all 20 targets and under-five-minute mission timing remain
unverified for the current all-books implementation. The next physical trial
will be added to the records above after its terminal outcome is checked.
The [matrix runner](../tools/book_matrix.md) starts each requested book in a fresh
world and independently checks target identity, containment, source stability,
reported collisions, and process cleanup.
