# Column 4 / blue: acquired, stopped before lift

Visible local Gazebo trial `121f7ff7e8ad`, seed 101, 15 September 2026. The correct target book contacted both fingers and passed the pre-lift retention check. The mission then aborted with `pick_recovery_failed` before an initial lift was observed. **No delivery or sub-five-minute completion was demonstrated.**

The solution ran for **346.669 s** from its reported launch origin to terminal summary; mission initialization to summary was 344.070 s. The journal recorded **zero collision episodes**. These are this run's recorded outcomes, not a certificate covering unobserved contacts. The run review confirms unchanged trial sources and that all owned processes stopped.

## Exact event sequence

Times below are ROS simulation seconds; the duration above is wall time.

| ROS time | Evidence |
| --- | --- |
| 108.152 | Planned front `[0.650858, -0.051809, 0.927929]` m in `base_footprint`; grasp advances 25 mm into the book. |
| 119.458 | Final open approach endpoint accepted using measured joint feedback. |
| 120.474 | Target identity latched: `book_col_5_row_4_blue`. The printed column / detected row mapping is column 4 / row 3. |
| 120.658 | Both fingers contact that target; measured gripper master position 0.0182116 m; stock named-contact acquisition accepted. |
| 121.280 | Three prospective lift/withdrawal legs pass the geometry recheck using the measured aperture. **This is geometry validation, not executed lifting.** |
| 121.478 | Bilateral contact, plausible width, settled grip, and retention verified before initial shelf lift. |
| 121.484–121.500 | Failure, abort, cancellation, and a published gripper hold. |

`pressure_qualified` was false; the accepted closure used `stock_public_position` / `stock_named_contact_confirmed`. The raw simulated contact-force values in the exact events are not a calibrated force or retention certificate.

## What the independent poses show

The read-only observer collected model poses separately from joint feedback and sent no data to the controller. Across **346 unique world-pose samples from 34.190 to 121.336 s**:

- The book center moved at most **0.7281 mm**, predominantly sideways along world Y. Its final displacement was `[+0.0097, +0.7280, +0.0009]` mm.
- Center-height variation was **0.0102 mm** peak to peak. The lowest corner remained between 0.791844 and 0.791858 m in the world frame.
- Maximum attitude change was **0.1693°**. There is no observed shelf lift, withdrawal, or dropped book in this recorded interval.
- The final world-pose sample precedes abort by 164 ms. This record does not establish what happened after that sample or after process cleanup.

The small final book movement is consistent with centering during closure. It does not demonstrate retention under transport or a rigid attachment to the gripper.

## Front alignment and joint FK

During the stationary reacquisition interval, the observer-derived front-face center was approximately `[0.650516, -0.052196, 0.916858]` m in `base_footprint`. The planned camera-derived front target differed by approximately **`[+0.342, +0.386, +11.071]` mm**. Its signed distance to the observed front plane was only **0.342 mm**. This supports the correct front-face/depth association; the visible-face center was about **11 mm above the geometric box midpoint**.

The tracker constructs its center by projecting the image rectangle center onto the fitted face plane. A visible-face center and the complete box midpoint are different quantities; these logs do not isolate the cause of their offset. The retained perception status contains quality and timestamp metadata, not the complete camera-space face corners/normal. Consequently, this evidence does not independently validate the complete estimated camera frame or camera calibration.

At bilateral close, forward kinematics (FK: the hand pose calculated from measured joint angles) put the grasp frame approximately **25.19 mm inside the observed book-front plane**. The book center was approximately **`[55.80, -0.007, -3.01]` mm in the grasp frame**. Measured-joint FK agreed with planned-joint FK to less than 0.001 mm in translation and 0.000002 rad in rotation at closure and the later pre-lift checks. These small numerical differences show settled joint tracking against this model; they are **not an independent measurement of hand-link accuracy or passive finger deflection**. The journal's `actual_grasp_position` and `actual_grasp_orientation` are planned-joint FK, not achieved-pose measurements.

The 11 mm center offset also affects how an estimated attachment box and initial shelf support are interpreted. No offset or threshold was changed for this analysis. The acquisition, FK agreement, and prospective geometry checks together do not constitute a physical attachment, floor-clearance, or complete shelf-collision certificate.

## Frames and timestamps

[Frame audit](frame_audit.json) records an offline conversion of the trial's exact URDF to SDF. The model root and `base_footprint` coincide. `base_link` is 76.2 mm above `base_footprint`; that fixed transform is already included in FK. Applying it again would introduce an erroneous height offset. World poses were transformed through the robot model pose; odometry coordinates were not substituted for world coordinates.

The book model, link, and collision-box origins coincide. The local box dimensions are 0.25 × 0.02 × 0.16 m. Its spawned pitch is π/2, so the front is the local **−Z face**, 80 mm from the center; local X is the book's height axis.

Model-to-model comparisons use one Gazebo pose timestamp. The hand/book comparisons use separately stamped joint feedback: at closure, pose 120.582 s versus joints 120.654 s; at retention, pose 121.336 s versus joints 121.474 s. No interpolation or simultaneity assumption is hidden. The arm was near stationary at those checks, but neither exact cross-stream synchronization nor live link deformation was measured.

## Files and reproduction

- [Run review](run_review.json) and [mission summary](trial_121f7ff7e8ad_summary.json).
- [Selected exact journal events](selected_journal_events.jsonl), [selected exact pose records](selected_pose_records.jsonl), and [selected perception events](selected_perception_events.jsonl).
- [Derived full-log analysis](full_pose_analysis.json) and [derived retained-subset analysis](selected_pose_analysis.json). Both produce the same book-motion extrema; the full log contains 346 unique stationary-window poses, the retained subset 16.
- [Provenance](provenance.json): trial source hashes, original-log hashes, and local run path. The original 1.4 MB pose stream and large source manifest remain under ignored `results/` and are not included here.

From the repository root, with the project's NumPy/SciPy dependencies installed:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/evidence/local_gazebo_20260915/column4_blue_grasp/analyze.py
```

The script makes no ROS or Gazebo connection, verifies the recorded FK/model source hashes, and computes the retained-subset result. To reproduce the complete locally preserved stream, add:

```sh
--full-pose-log results/matrix_20260915T171649_729702Z/01_seed101_column4_blue/pick_pose_diagnostics.jsonl
```

The optional full log must match its recorded SHA-256. Reproduction was checked against the current unchanged model/FK sources for both inputs.
