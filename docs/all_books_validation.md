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

## Current implementation

- Books-mode observations retain the selected numbered bay, confirmed row,
  observation timestamp, and fresh command epoch. Close-view detections must
  agree with that context.
- Depth fitting can explain a connected front face plus attached orthogonal
  side/top faces while retaining fit uncertainty and geometric consistency
  requirements. The captured failed blue view and synthetic four-colour,
  four-row views pass; corrupted geometry remains rejected.
- Lower-shelf planning must admit a complete empty approach, lift, supported
  carry, and recovery before dispatching motion. The second and third rows use
  distinct torso and wrist routes. Bottom-row entry is under development.

The latest software checks do not yet establish delivery for all 20 targets or
under-five-minute timing. Physical trial results will be added as they finish.
The [matrix runner](../tools/book_matrix.md) starts each requested book in a fresh
world and independently checks target identity, containment, source stability,
reported collisions, and process cleanup.
