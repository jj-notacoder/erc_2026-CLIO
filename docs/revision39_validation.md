# Revision39/40 validation history — completed Run38

Revision40/final07 was the 268-file software snapshot used by Run38. Its 2,667-test
certificate and revision39 geometry history below remain specific to those
sources. The [current revision41 record](revision41_validation.md) contains
final08 validation, completed Run38 evidence and Run39's physical delivery
and sampled-geometry limits.

Run37 picked up and carried the book, then reached its automatic trial limit
during PLACE planning, before any placement motion or release. Its
[completed outcome](revision38_validation.md#physical-outcome) motivates clearer
stage timing and less repeated collision work. It does not establish that
collision checks failed or that the configured planning deadline caused abort.

## Inherited revision39 changes and focused evidence

Planning diagnostics record stage, elapsed time, search counts and the selected
planning inputs, and retain cancellation/failure details. They add visibility
without changing motion admission or the search deadline.

The static-pair helper tries hull-derived directions, but tests separation
using **all original vertices at the current pose**, including compound links.
A gap must exceed 100 micrometres plus a conservative arithmetic allowance.
Touching, overlapping and uncertain cases keep the original triangle/containment
fallback; custom surfaces and moving-left pairs bypass the helper. Cached
directions are hints, never reusable collision verdicts. This may safely remove
legacy numerical false positives, including degenerate facets; it does not
promise universal Boolean equivalence or physical clearance.

[Focused evidence](evidence/revision39/static_pair_separation_summary.json)
records **150 passing tests** (33 new and 117 existing cases), with no failures,
errors or skips. These are component/caller checks, not the combined full suite.
One matched instrumented seven-pose workload with changing recorded parked-arm
contexts used **28.06% less checked wall time** (1.093553 to 0.786713 seconds).
The frozen-context total was **4.55% higher** (0.758562 to 0.793081 seconds),
including first helper/SciPy import; warmed frozen samples were essentially
unchanged. Inputs, verdicts, pair order and existing cache counts matched.

This baseline-then-candidate comparison uses Run36 planned poses and Run37
recorded context, includes instrumentation, and is not a full planner, live
mission or guaranteed speedup. Official assets/physics, model margins, sampling,
contact checks, selected settings and search limits are unchanged.

## Full-suite regression and revision40 correction

Final06 ran **2,655 tests: 2,654 passed and one failed**, with no errors or skips.
Selected/default constructors, official checks and unchanged-source checks
passed, but the overall snapshot **failed validation and was not certified**.
The [unaltered failed result](evidence/revision39/final06_failed_validation_result.json)
is retained. Eager construction of initial diagnostic arguments read a missing
held-book field before the logger's error handler, replacing the original
required-bin admission error. The earlier source review missed this runtime
evaluation-order issue; recovery of an AST after removing logging was insufficient.

Revision40 protects the three telemetry boundaries while allowing original
admission, geometry and solver errors to propagate. **All 26 focused checks
passed**, including 14 unchanged registered-bin cases and 12 new diagnostic
boundary cases. The [correction summary](evidence/revision40/telemetry_fix_summary.json)
records the independent review, unchanged non-telemetry geometry/control AST
and identical valid initial payload apart from elapsed time. At initial final07 certification, the older nominal route was not rerun on
revision40; its applicability rested on that narrow reviewed change. A later
Run38 logged-input replay ran on exact final07 source and is documented in the
[current record](revision41_validation.md#independent-recorded-input-checks).

## Executed nominal placement replay

<!-- REVISION39_REPLAY_BEGIN -->
**PASS on revision39 in 211.494 wall seconds.** The unchanged complete planner
returned 12 setup legs, 11 Cartesian waypoints, 54 opening checks to .069 m and
12 direct empty-return legs. Its first candidate used 86 IK calls, 2,338 scene
samples and 110 exact-cache hits. The [portable nominal summary](evidence/revision39/run37_nominal_route_summary.json)
pins the executed result, source and stage timing. This successful geometry
experiment does not turn the separate failed final06 full suite into a pass.

The fixture uses the same-entry published bin/table scene at 276.046 ROS seconds
and atomic joints at 276.094 (+48 ms), original nominal PICK attachment and
measured carry as a replacement IK seed. The exact live consumed scene and stored
staging were unlogged. Right/head/master context was held fixed. No evaluator
poses entered planning, and no mismatched camera frame was refitted.

All sampled loaded setup, Cartesian, nominal opening and empty-return checks
ran with unchanged margins and support/IK limits. The 420-second experiment cap
did not fire; the internal selected search allowance remained 900 seconds.
Five official description files and 40 pinned mesh files were rehashed, reusing
the earlier 895-file source proof rather than claiming a new full asset audit.
This establishes fixture-specific nominal feasibility, not the route consumed
by Run37, live planning speed, continuous clearance, actual passive deflection,
attachment rigidity, fall/landing behavior or physical delivery.
<!-- REVISION39_REPLAY_END -->

## Historical final07 software validation and Run38 outcome

<!-- REVISION40_FULL_VALIDATION_BEGIN -->
**Revision40 / final07 passed all 2,667 full ROS tests**, with zero failures,
errors or skips. Actual selected/default constructors passed for manipulation,
mission, perception and navigation; no executor or action goal ran in those
constructor checks. All 268 package files remained unchanged. The 895-file
official check passed (893 exact files plus two audited bytecode-header
refreshes with unchanged payloads), and all 40 resolved URDF meshes were checked.

The [exact exported certificate](evidence/revision40/software_certificate.json)
is included byte-for-byte. The validation completed at 15:52:13 UTC on
11 September 2026 in 271.908 wall seconds. The selected profile remains unchanged,
including its explicit 900-second planning allowance; ordinary defaults retain
420 seconds. The record certifies these software/configuration checks, not
physical delivery or mission timing. Its export-time promotion/dispatch flags
remain unchanged and do not describe later trial lifecycle.

Certificate SHA-256:
`e96906f690ff6dcf25d4d69cb8bfb4b26b6de015cf49b2f98812e268ee3d59d3`.
The source tree, snapshot, profile and constructor hashes are in that certificate
and the [validation-linked correction summary](evidence/revision40/telemetry_fix_summary.json).
<!-- REVISION40_FULL_VALIDATION_END -->

<!-- RUN38_FINAL_OUTCOME_BEGIN -->
**Run38 completed: automatic placement_failed before any PLACE motion.** Full
nominal candidate checks passed at 1383.113 planning wall seconds, then the
post-candidate search deadline rejected 1316.044 seconds against 900 allowed.
The book had been picked and carried; no placement motion, opening, release or
empty return followed. The [completed portable record](revision41_validation.md#completed-run38)
binds the 360-file archive, 268 matching source files and exact terminal events.
<!-- RUN38_FINAL_OUTCOME_END -->

The [profile guide](collision_quality_profile.md) retains the explicit
`profile:=collision_quality` command and development time allowances. Ordinary
defaults remain distinct. Collision-free delivery, five-minute timing and varied
reset trials remain to be demonstrated.
