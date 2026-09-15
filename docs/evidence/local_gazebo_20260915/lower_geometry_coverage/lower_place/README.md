# Lower-row PLACE geometry probes

These are **synthetic mixed-recorded fixtures**, not live lower-row placements.
They combine row-specific complete offline PICK outputs with the exact selected
bin/table, parked right/head, base and gripper-master values from visible blue
trial `105d91426ddd`. No simulator ground-truth pose is a planner input.

| Fixture | Outcome | Wall time | IK calls | Complete candidates |
|---|---|---:|---:|---:|
| Bottom, saved `3:green` lateral case | Failed existing IK limit | 16.203 s | 512/512 | 0 |
| Row 2, saved guarded nominal case | Passed full PLACE preflight | 53.457 s | 141/512 | 3 |

Bottom's default negative-wrist search rejected 145 IK attempts, eight
nonnegative-wrist poses and 72 joint steps; it exhausted its budget before a
complete candidate. No full opening or release endpoint was admitted.

Row 2 checked 5,000 exact scene samples and 12 setup legs. The third complete
candidate passed full body/tool/book/bin/table geometry, signed support,
opening, static open-hand endpoint and ordinary release-only identity checks.
The first two complete candidates were rejected by the existing guards.

Both fixtures first verified the official dimensions `[.16,.02,.25]`, then
reconstructed the saved attached corners with current production code. Both
matched exactly, with maximum absolute difference zero. The row-2 PICK input
comes from an earlier saved guarded fixture; this does not revalidate row-2
PICK against every subsequent source change. Bottom input comes from the new
complete current-source lateral probe.

The scene coordinates and all registration/material margins are unchanged.
Torso target is the actual profile's `.35` m. Each row's target is regenerated
by `book_centered_place_target`; the original negative-wrist solver policy is
used with no registered target override. Runtime IK limit 512, wall budget
1,800 s, position/orientation tolerances `.002/.02`, maximum joint step `.4`,
joint margin `.1`, and support threshold `.75` remain unchanged. Each separate
serial invocation had an external 180-second diagnostic alarm; neither reached
it. No solver, scene predicate or production source was replaced.

The first bottom invocation stopped in 0.074 s before search because the harness
mistook `cached_plan.requires_gravity_support=False` for final transport mode.
That flag describes initial vertical-pinch entry. Production enables supported
mode during the cradle roll. The corrected harness preserves the initial flag
and requires the ordinary static supported-transition predicate at the saved
final terminal before starting PLACE. The original failure is preserved in
`bottom_fixture_admission_initial.json` as a fixture-compatibility record.

Both processes exited; a post-run `/proc` scan found no replay or geometry-worker
processes. There were zero controller calls. Full package source hashes remained
unchanged through each run. Inputs, plans/rejection diagnostics, stage events,
and hashes are in the JSON/log files and `summary.json`.

Frozen logical timestamps allow the actual freshness/admission functions to run
inside the offline fixture. They do not establish live freshness. The blue
master-aperture value is an explicit mixed-fixture assumption, not measured
retention for either lower book. Results do not establish physical release,
containment, falling-book dynamics, or complete mission time.
