# Positive PLACE entry restoration and stationary diagnostics

Promoted change: three runtime files in `erc_phase1_solution/`, and two new tests in `test/`.
`source_manifest.json` pins the current production base and the reviewed draft.
`runtime.patch` contains only those three production deltas. The archived patch records the three runtime changes; tests live in the source package.

## Observed failure

Live trial `105d91426ddd` on source `0a5fa41` completed PLACE planning and torso
raising, then executed the first bin-transition arm leg. The qualification before
leg 1 exhausted its unchanged 500,000,000 ns ROS settling window (2.916432732 s
wall time). Neither stationary producer window was established. The older event
lacked the specific failed joint/velocity predicate, so the exact physical cause
is unknown. Planning and torso motion did not consume that fresh settling window.

## Conservative change

For the actual accepted registered PLACE plan with measured positive carry wrist,
restore only its first `.8 s` bin-transition leg to exactly `800,000,000 ns`.
Ordinary cap 2 plus additional factor 2 otherwise shortens that leg to `.2 s`,
subject to the existing velocity minimum. The immutable accepted-plan signature,
full route, scene/attachment/contact identity, positive measured wrist, exclusive
owner, original duration, and exact publication goal bind this restoration.
Both fresh locked velocity admissions and the existing 80% velocity headroom
check remain required. A direct sender cannot omit those admissions. All other
leg durations and the negative/default path remain unchanged.

The integer retimer floor only increases the existing 87,500,000 ns minimum and
cannot exceed the original nominal watchdog allowance. It does not create a new
motion proposal or alter positions.

Qualification keeps its original .5 s ROS limit, 3 s progress watchdog, 100 ms
joint/odometry stationary windows, freshness/gap checks, position/velocity bounds,
retention checks, and final locked publication check. New telemetry records the
already evaluated predicate result, aggregate maximum errors/velocities, and at
most 12 existing raw samples. It adds no sensor subscription, motion, or wait.

This is a conservative physical hypothesis, not a proven settling fix. The next
live trial must establish whether restoring duration suffices; bounded diagnostics
will identify the failed raw predicate if it does not.

## Validation

`focused_tests.log`: **145 passed in 4.93 s**, using actual production sender,
executor, and PLACE caller methods with controlled feedback, and the existing
stop/retimer regressions. New tests cover exact 800 ms serialization, unchanged
other timings, direct-sender bypass rejection, identity/route/freshness vetoes,
nullable negative dispatch, bounded diagnostics, and unchanged qualification
windows. This test result makes no physical controller claim.

Independent read-only review approved runtime hashes 212a387b / 5178ab9f /
5785825b. The reviewed bytes were promoted and the historical source inverse was regenerated.
The production rerun passed **310 tests in 11.92 s**, including release-only
planning/timing historical regressions, PLACE dispatch/flow, stop and retimer checks.
