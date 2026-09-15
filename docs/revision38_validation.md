# Revision38 validation record

The selected `collision_quality` development profile passed software validation
and a complete nominal placement replay. **Run37 completed with an automatic
trial timeout before placement motion.** The [profile guide](collision_quality_profile.md) gives the tested
`profile:=collision_quality` command. The ordinary column/colour-only command
still selects legacy defaults and does not reproduce that configuration.

## Evidence included in this repository

| Evidence | Result | What it establishes |
| --- | --- | --- |
| [Exact final05 software certificate](evidence/revision38/software_certificate.json) | 2,622 full ROS tests; zero failures, errors or skips; selected/default four-node constructors passed; all 262 source files unchanged | Software and configuration checks on the identified snapshot; 895 official files and 40 resolved URDF meshes checked |
| [Nominal route summary](evidence/revision38/nominal_route_summary.json) | PASS in 213.790 wall seconds; 86 IK calls; 12 setup legs, 11 Cartesian waypoints, 53 opening samples and 12 empty-return legs | Sampled nominal robot/tool/book, screen, table and bin checks through the complete planned route, opening and return |
| [Contact aggregation summary](evidence/revision38/contact_aggregation_summary.json) | 35 focused cases; all 2,000 equal-input deliveries matched; median callback CPU .928398 to .494719 seconds, 46.7% less | Equivalent saved-gripper callback results with lower measured component cost; no demonstrated whole-mission speedup |

The certificate is an unchanged copy of the completed software record. Its
`production_promoted: false` and related flags describe its export stage, not
subsequent trial lifecycle. The two summaries are explicitly labeled selections
from retained results; they preserve the original result/source hashes and
limitations. They are readable without the large raw archive, but are not a
self-contained raw-data reproduction kit. No tests or geometry were rerun to
prepare this documentation.

The certificate SHA-256 is
`c12d024cede84c058d73b9a97cccbfdfecc8a9e02a7f0424d79385f2dc7a88c7`.
The selected runtime-profile hash is
`16b95a58194d23225d3775b8cd07afa3c72317ce112895082aa8690a018e56ef`;
this identifies the archived runtime overrides, not the installed YAML file's
bytes. The official runtime remains pinned to
`b1f9b05e20f4750b59f3321d88f172cc4dbb1386`.

## Scope of the revised path

Placement uses the onboard fitted bin floor-center and orientation for both
the target and collision material, with the table fit from the same depth
epoch. The legacy navigation point is preserved. The path checks the head-screen
visual envelope, table/bin material and nominal held/open gripper geometry;
measured endpoints and fresh context precede the next checked command. Observed
arm/tool contact with the bin or table latches a stop through release and return.

The successful replay used recorded Run36 RGB-D/TF at 300.136 ROS seconds and
atomic joints at 300.290 seconds: a disclosed 154 ms retrospective skew, not a
test of fresh live-epoch admission. It used the original nominal attachment and
measured carry as an IK seed because exact live staging was unlogged. There were
no evaluator inputs or counterfactual base shifts. Ten proximal steps expanded
to 12 checked setup legs. The older 22-leg result used a different Run33 fixture
and a 100 mm counterfactual backoff; those counts are not a controlled speed
comparison. Every loaded/opening/return gate and bounded search limit remains.

Nominal sampled geometry does not establish actual passive-finger deflection,
attachment stability, released-book dynamics, continuous collision freedom or
physical delivery. The callback comparison covers the saved gripper-message
component under a controlled fixture, not all contact traffic or a concurrent
planner. Official physics and executor settings were unchanged.

## Physical outcome

<!-- RUN37_FINAL_OUTCOME_BEGIN -->
**Run37 (`d9d1b6fd445f`) picked up and carried the book, then automatically
aborted on `trial_timeout`.** The configured 2700-second trial limit produced
ABORTED at 547.300 ROS seconds / 2703.059 observer wall seconds. This was not an
operator cancellation. The worker's subsequent `scene_cartesian_cancelled`
event followed that cancellation and does not prove geometric infeasibility.

No PLACE `ik_ready`, placement motion, opening, release or empty return occurred.
The final book remained held outside the bin; the bin did not move and no
relevant robot/table/bin or target-book/bin contact was reported. Held possession
did not establish a rigid attachment: the sampled planning hold plus observer
tail reached **17.412 mm / 24.181 degrees** of book motion relative to arm7.
These evaluator measurements were used only to assess the outcome.

The [portable outcome summary](evidence/run37/outcome_summary.json) records the
failed and unexercised phases, 1,279 pose samples and archive checks: all 355
archived files and 262 package files matched; the unchanged official check
passed for 895 files. Sparse negative collision samples are not continuous
clearance proof. The [revision39/40 history](revision39_validation.md)
records final07 and completed Run38. The [current revision41 record](revision41_validation.md)
contains final08's 2,668-test pass and Run39's completed physical delivery,
with sampled geometry and explicit limits.
The earlier final05 certificate above remains unchanged.
<!-- RUN37_FINAL_OUTCOME_END -->

Historically, Run33 and Run34 delivered the book but had collisions, taking
23 min 05.962 s and 25 min 47.556 s to observer `DONE`. Run34's final footprint was
inside and floor-supported; its strict core flag was false due solely to a
0.207 micrometre floor residual. Run36 picked up and carried the book, then was
operator-cancelled during placement planning; release/empty return were
unexercised. These outcomes are separate from revised-route validation. See the
[historical record](official_current_validation.md).

Collision-free delivery, the five-minute unedited video requirement, varied
reset trials and reproduction through the required default launcher remain to
be demonstrated.
