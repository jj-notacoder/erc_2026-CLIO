# Revision41 software and Run39 physical delivery

Revision41 / final08 passed **2,668 full ROS tests**, with zero failures, errors
or skips, actual selected/default constructors for all four team nodes, and
unchanged source/asset checks. The [exact certificate](evidence/revision41/software_certificate.json)
and [selected profile](evidence/revision41/selected_runtime_profile.json) are
included byte-for-byte. Run39 completed a measured delivery and empty return,
with the limitations below. Earlier
revision39/40 evidence and the failed final06 snapshot are preserved in the
[historical record](revision39_validation.md).

## Completed Run38

Run38 (`c0c9e18c32f6`, final07) picked up, extracted and compacted the book,
returned to the bin and obtained registered onboard bin/table geometry.
During PLACE planning, the full candidate validator passed **2,337 scene
samples**, including loaded setup, Cartesian legs, full opening and empty
return checks. `candidate_accepted` was recorded at 498.086 ROS /2354.141
observer wall seconds, **1383.113 planning wall seconds** after planning began.
Those opening/return checks were geometric predictions, not robot operations.

The search's next deadline check reported **1316.044 search wall seconds**
against its **900-second** allowance and rejected the otherwise checked plan.
Mission ABORTED automatically with `placement_failed` at 498.200 ROS /
**2354.877 observer wall seconds**. No PLACE motion, gripper opening, release
or empty return was executed. This was not an operator cancellation or the
later container shutdown. The observer finished at 506.544 ROS /2406.277 wall
seconds. These wall durations are distinct from simulated ROS time.

The [portable outcome](evidence/run38/outcome_summary.json) binds the recorded
events and the separately retained archive: **360 evidence files and 268
matching source files**, with unchanged official assets. The trial summary
reports zero collision episodes and no target/bin contact; that is not a claim
of continuous collision clearance. No physical delivery occurred.

## Independent recorded-input checks

The [full nominal replay](evidence/run38/nominal_logged_replay_summary.json)
passed on exact final07 source in **189.318 monotonic seconds**. It used the
actual logged staging seed, carried/torso vectors, target points, rotation,
nominal attachment, master and selected scenes. A nearby atomic parked
right/head sample was held fixed, **156 ms after** the initial planner event.
The first candidate passed 12 setup goals, 11 Cartesian solutions, 54 opening
samples and 12 empty-return goals, with 23 IK calls. No alternative seed or
profiling run was performed. It preserves the 900-second internal allowance.
This verifies that fixed nominal fixture, not the live callback/cache stream,
physical deformation or a controlled live speedup.

The [screen audit](evidence/run38/screen_audit_summary.json) found no sampled
overlap in **125 PICK +53 COMPACT atomic samples**. Separately stamped actual
tool/book comparisons were possible for 177; one PICK sample lacked a pose
stamp. Both-arm visual/collision meshes, nine nominal tools and available
actual tool/book poses were compared with the URDF visual screen envelope.
Maximum sample gaps were .960/.388 seconds and pose skews stayed within 14ms
where available. This is not general robot self-collision, table/placement or
continuous-clearance proof, and the visual screen is not a physical contact
sensor.

## Revision41 allowances and validation

The selected search allowance increases from 900 to **1800 wall seconds**.
The validation ceilings permit that value; the ordinary default stays 420.
Search still shares one original deadline and keeps 512 IK calls, 24 retained
paths and 6 full candidates. The selected Run39 mission allowance is **5400
wall seconds**, explicitly passed by the launch harness, with camera/observer
caps 5700. The manipulation command and mission manipulation-wait limits remain
2400. These are development allowances, not faster motion or compliance with
the five-minute unedited video requirement. Geometry predicates, official
physics, stock .017 grasp and fitted placement target are unchanged.

The full suite completed at 18:22:29 UTC on 11 September 2026 in 261.143 wall
seconds. All 268 package files were unchanged during validation. The official
895-file checks passed, including 893 exact files and two audited bytecode
header refreshes with unchanged payloads; all 40 resolved URDF meshes passed.
Actual selected/default constructors exercised 1800/420 planning allowances
without spinning an executor or dispatching robot goals.

Certificate SHA-256:
`79995071b49f46d980c42bbd28dbba0454d2053197cd04bb45eb8ff41e67f6e0`.
Source tree:
`7ba12ae482dd1ef5a25f9dbb12df0692d678c9dc84bca758ee0f8ffbd08fa29b`.
Profile:
`590bc458a136cd9d2ab9073104cd496077e5b2df65f0079ed4010a878dedd705`.
Certificate export-time promotion/dispatch flags and the profile's pre-validation
description remain unaltered historical metadata; they do not describe a later
physical trial outcome.

## Completed Run39 physical result

Run39 (`3f9468467cf5`) autonomously picked, extracted and compacted the book,
returned to the bin, completed PLACE planning and motion, opened the gripper,
and completed the empty arm/torso return. `DONE` was recorded at **516.600 ROS /
2432.730534 observer wall seconds (40 min 32.731 s)**. These time bases are
distinct, and the wall duration does not meet the five-minute unedited video
deliverable. This is one selected-profile development trial, not a success-rate
estimate or competition score.

The [portable physical outcome](evidence/run39/outcome_summary.json) binds the
retrospective evaluator analysis and independently verified **365-file archive**,
all **268 source files**, 895 official-file checks and 40 recorded mesh pins.
All eight stock-book corners are inside the conservative cavity core in the
final **36 identical poses, 494.668–525.694 ROS**, spanning 31.026 ROS /75.439
observer wall seconds. Floor residual is **+0.244075 micrometres**; the separate
1-micrometre floor-support diagnostic agrees. Minimum side clearance is only
**0.474768 mm**, with 88.793977 mm along the bin length. The side margin is an
observed near-wall landing, not a robustness or repeatability guarantee.

The only saved scoped contact episode is the intended **target book/bin** pair
at 491.542 ROS. The jaw's sample 2 ms earlier was already 0.068985119 m; measured
fully open was verified at 491.698 ROS. No saved arm/tool table/bin pair or
payload hazard occurred, and the bin's sampled pose stayed unchanged. Ten
Cartesian return endpoints plus arm HOME and torso HOME were measured; the
open hand's final HOME event was 516.402 ROS. `DONE` and those operation events
are not themselves containment proofs; evaluator poses were used only for
retrospective assessment and never as controller inputs. No gentle-placement
score is asserted from geometry, contact or the final settled pose.

The [actual sampled geometry](evidence/run39/actual_geometry_summary.json) found
no overlaps in **313 atomic screen states** across initial stow, PICK, COMPACT,
head/bin approach and physical PLACE/return, and **124 eligible table/bin states**
during head approach and PLACE. There were 310 eligible actual-tool/book pose
comparisons; two excessive-skew samples and one missing-pose sample retained
their atomic screen checks but omitted actual-pose/scene comparisons. Maximum
selected phase gap was 1.012 ROS seconds. Smallest arm visual-screen AABB
separation was 7.165 mm; during PLACE, arm visual/table was 31.781 mm and
actual-tool/bin was 16.585 mm. These conservative sampled bounds do not prove
continuous clearance, all robot self-pairs or enabled physical screen contact.

Planning accepted one nominal candidate after **1294.107 planning wall seconds**,
with 2336 scene samples and 23 IK calls; search used **1229.878 of 1800 seconds**.
Before motion, the book shifted by up to **17.254 mm /20.742 degrees** relative
to arm7 over the stationary hold, while arm7-base translation stayed within
36.209 micrometres. A separate earlier eight-corner comparison already found
15.429 mm protrusion outside the padded nominal attachment at 327.588 ROS.
Those are distinct measurements: the final hold was not refitted to a new
attachment. Successful delivery does not validate the old envelope, and later
handoff-guard/AABB proposals remain separate, unpromoted candidates.

The measured first setup segment added no more than 0.961 mm /0.390 degrees
of book-relative-hand movement through 483.966 ROS. Deliberate later opening
and withdrawal are not grip slip. This physical result improves on Run33/34's
recorded scene collisions, while long planning, attachment drift, the narrow
landing margin, untested resets and submission preparation remain unresolved.

Host continuity is tracked separately: the Run38 container was already stopped
with exit 255 when post-trial shutdown was attempted; its cause is unproven in
that record. A bounded foreground WSL lease was subsequently prepared for
development-session continuity. It does not explain or change Run38's earlier
recorded planning-budget rejection.
