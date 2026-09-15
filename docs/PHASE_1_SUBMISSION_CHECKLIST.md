# Phase 1 Submission and Recording Checklist

Use this as the final gate for the Emirates Robotics Competition 2026 Phase 1 simulation submission. The supplied rules give a Phase 1 deadline of **15 September 2026**. Recheck organizer announcements and the submission form before uploading.

This checklist does not contain final-trial results. Fill the final-trial fields only from recorded runs on the exact frozen simulator baseline.

Current development boundary: retry 19 (`2172a3aa1379`) is the terminal same-seed engineering diagnostic for seed 101, column 2 and red, not one of the required five randomized official trials. It correctly identified column 2 and the red book in row 1. Shelf navigation measured 1.444066 m planned, 1.407512 m executed, ratio 0.974687, 10.750 sim s and 44.561 mm end error; book-standoff navigation measured 0.770540 m planned, 0.730692 m executed, ratio 0.948286, 9.150 sim s and 44.507 mm end error. IK selected `loaded_clearance_index=1`, with 5 extraction, 3 lowering and 2 HOME waypoints. Bilateral hold verification passed at 0.018000433 m, and retention survived all 5 extraction plus 3 lowering legs. On the first home-transition leg (leg 8), the red book contacted `arm_left_5` at ROS 93.552 s; retention monitoring reported `grasp_lost` at ROS 95.714 s. Safe carried recovery succeeded, then visual reacquisition failed. That run predated the current carried-book-versus-robot mesh checker and compact upright transport sequence. The new host regression permits the lowered extended pose only for a 0.70 m fixed-heading shelf retreat, then requires a 0.436 m radius-gated compact posture before turning and filters bin IK through the loaded sweep. These changes still need live Gazebo validation. The run terminated with `success=false` and `book_reacquisition_failed` after 551.510 wall s, four navigation goals, one pick attempt and one collision episode. No retained verified grasp, return, bin detection, placement or mission success is claimed. The package is not competition-ready; do not copy retry 19 values into the final-trial table.

## 1. Team identity and ownership

- [ ] Replace `TEAM_NAME` everywhere with: `____________________________`.
- [ ] Replace `UNIVERSITY` everywhere with: `____________________________`.
- [ ] Put the team name visibly on screen before robot operation begins.
- [ ] Confirm the submitted code and media are the team's work and contain no credentials, private keys, personal paths or unrelated files.
- [ ] Record the final commit SHA: `________________________________________`.

## 2. Freeze the evaluation environment

- [ ] Obtain written confirmation of the simulator release/tag or image used by the evaluator: `________________`.
- [ ] Resolve the rules mismatch: the supplied Phase 1 PDF names `v1.0.0`, while this development workspace uses official `v1.0.3`.
- [ ] Note that official `v1.0.3` fixes book grasping and introduces `_raw` gripper controllers behind public clamp-proxy topics.
- [ ] Confirm the package still commands `/gripper_left_controller/joint_trajectory`, not the version-specific `..._raw` controller.
- [ ] Use a supported Linux x86_64 host with X11 and Docker Compose v2.
- [ ] Confirm approximately 15 GB of free disk space and sufficient recording space.
- [ ] Compare the frozen checkout with official tag `v1.0.3` and resolve every simulator or Docker diff before recording. Current development diffs add evidence bind mounts and an entrypoint comment; restore them or obtain written organizer approval. Do not modify robot kinematics, collision geometry, physics or the competition world.
- [ ] Save `git status --short`, the chosen simulator tag and the image/container identifier with the trial record.

## 3. Build gate

On the host:

```bash
./docker/up.sh --build
./docker/attach.sh
```

Inside the container:

```bash
cd /opt/erc_ws
colcon build --symlink-install
source install/setup.bash
```

- [ ] Build completes without errors.
- [ ] `ros2 pkg prefix erc_phase1_solution` resolves to the current workspace.
- [ ] `ros2 pkg executables erc_phase1_solution` lists all four nodes.
- [ ] `ros2 launch erc_phase1_solution solution.launch.py --show-args` lists `shelf_column_number`, `book_colour`, `team_name`, `rviz`, `trial_timeout_seconds`, and `dry_run`, with timeout default `270.0`.
- [ ] The final `config/solution.yaml` is archived with the submission evidence.
- [ ] Run `python3 -m pytest src/erc_phase1_solution/test -q` and record `FINAL_TEST_PASSED/FINAL_TEST_TOTAL` passed in `FINAL_TEST_DURATION` on `FINAL_COMMIT_SHA`.

Final development validation: 139 tests passed in 11.59 s; Python bytecode compilation was clean; `git diff --check` exited 0 with line-ending warnings only; and `colcon` built one package in 4.97 s. Production gates ignore right-gripper contacts for the left pinch, refuse further closing after an undersized pinch, enforce a 0.018 m hold with a 0.016 m operational floor, and abort on an ambiguous held-payload state. The subsequent 2026-09-04 host-only geometry/helper regression passed 118 tests with two ROS-dependent modules skipped after adding carried-volume checks against the official base, torso, head/camera and both arm-link chains, exact nonadjacent-link self-collision and closed-mesh containment checks at endpoints and adaptive interior samples, measured parked-arm geometry, a 60-degree absolute tilt ceiling, a 0.45 m navigation-radius gate, hard-stop IK margins, payload-filtered placement, a straight shelf retreat and complete four-row overview gating. The selected retry-19 compact route passed an earlier 601-sample exact triangle-surface audit; the containment-enabled route passed all production checks and a 100-sample dense spot audit. Its modeled base/torso/head/arm-link/book envelope is limited to 0.436 m, its book corners reach 0.418 m and the fold peaks at 42.43 degrees of tilt. Tool-link and gripper palm/finger geometry remains outside the generic runtime guard. These changes still require a clean ROS/Gazebo run and do not replace the final placeholders above.

## 4. Pre-recording simulator checks

Start the official simulation in the first container terminal:

```bash
ros2 launch erc_bringup simulation.launch.py
```

- [ ] Wait until Gazebo, the robot and all expected controllers finish loading.
- [ ] Confirm `/clock`, `/odom`, `/scan_front_raw`, `/scan_rear_raw` and onboard RGB-D topics are publishing.
- [ ] Confirm arm, head, torso and public gripper command interfaces exist.
- [ ] Confirm `/contacts` and `/bin_contacts` are available for evidence.
- [ ] Confirm no other process is publishing `/cmd_vel`.
- [ ] Confirm `ROS_DOMAIN_ID` does not conflict with another ROS 2 system (official default is `23`).
- [ ] Reset the competition world through the official procedure; do not manually arrange objects.
- [ ] Archive current development artifacts, then use empty final-run `erc_images/` and `results/` directories. Do not run the five-trial summarizer over the mixed development directory.
- [ ] Verify adequate free space and that the screen recorder captures Gazebo, both terminals and the timer legibly.

## 5. Recording requirements

The supplied rules require an unedited video no longer than five minutes.

- [ ] Start the recording before launching the simulation.
- [ ] Show `TEAM_NAME` and `UNIVERSITY` on screen before robot operation.
- [ ] Show the base simulation launch command and its startup.
- [ ] Display a continuously visible on-screen timer.
- [ ] Do not pause, cut, splice, speed up or otherwise edit the run.
- [ ] Keep both commands and relevant terminal output readable.
- [ ] Keep enough of Gazebo visible to judge navigation, collisions, grasp retention and placement.
- [ ] Stop the recording before it exceeds 05:00.
- [ ] Retain the original recording file and checksum.

In the second attached container terminal, run from the workspace root; outputs are written to the repository-level `erc_images/` and `results/` directories:

```bash
cd /opt/erc_ws
source install/setup.bash
```

The exact competition command to show and execute is:

```bash
ros2 launch erc_phase1_solution solution.launch.py shelf_column_number:=2 book_colour:=red
```

- [ ] Replace only `2` and `red` with the assigned column and colour.
- [ ] Do not add `rviz:=true`, `dry_run:=true`, `trial_timeout_seconds:=...`, or other development arguments unless organizers explicitly permit them.
- [ ] Capture the exact assigned input here: column `_____`, colour `____________`.

## 6. Live scoring/evidence checks

The maximum documented simulation score is 19 points; collisions deduct 0.5 points each. Treat the organizer/evaluator as the authoritative scorer.

- [ ] Correct column is published on `/erc/shelf_column_identification` (`std_msgs/msg/Int32`).
- [ ] A timestamped column-detection image exists and clearly shows marker bounding boxes.
- [ ] Robot reaches the correct shelf without collision.
- [ ] Correct row is published on `/erc/shelf_row_identification` (`std_msgs/msg/Int32`).
- [ ] A timestamped requested-book image exists and clearly shows its bounding box.
- [ ] The left gripper visibly closes on the requested book.
- [ ] Preserve the guarded close/hold stage, bilateral requested-book contact and primary finger-position evidence in every final trial. Retry 19 live-confirmed the 0.018 m hold but lost retention during home transition.
- [ ] The requested book visibly remains held during extraction and return; controller success and requested-colour contact alone do not prove exact selected-book retention.
- [ ] Robot returns to the start/bin area without collision.
- [ ] The red bin is visibly identified.
- [ ] The book is visibly placed inside the bin.
- [ ] Preserve the accepted raw `/bin_contacts` pair. Mission success requires classifier-accepted contact, but a generic collision name does not identify the exact book; verify requested-book identity from the unedited video.
- [ ] Placement is gentle if claiming the four-point gentle-placement score; otherwise record the observed two-point drop outcome accurately.
- [ ] Review `/contacts`, the video and mission logs for every collision episode.
- [ ] Preserve the planned and executed path display/screenshots for the report.

Useful read-only checks in an additional attached terminal:

```bash
ros2 topic echo /erc/shelf_column_identification
ros2 topic echo /erc/shelf_row_identification
ros2 topic echo /bin_contacts
ros2 topic echo /contacts
```

## 7. Five measured final trials

Run at least five reset, randomized trials on the frozen evaluator-approved baseline. Keep failures; do not select only successes. Development retries are not final-trial values and must not be copied into this table.

| Trial | Trial ID | Tag / seed | Assigned input | Success | Failure reason | Column | Row / correct? | Grasp retained | Bin contact | Collisions | Wall time | Observed score |
|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| 1 | `FINAL_T1_ID` | `FINAL_T1_TAG_SEED` | `FINAL_T1_INPUT` | `FINAL_T1_SUCCESS` | `FINAL_T1_REASON` | `FINAL_T1_COLUMN` | `FINAL_T1_ROW` | `FINAL_T1_GRASP` | `FINAL_T1_BIN` | `FINAL_T1_COLLISIONS` | `FINAL_T1_TIME` | `FINAL_T1_SCORE` |
| 2 | `FINAL_T2_ID` | `FINAL_T2_TAG_SEED` | `FINAL_T2_INPUT` | `FINAL_T2_SUCCESS` | `FINAL_T2_REASON` | `FINAL_T2_COLUMN` | `FINAL_T2_ROW` | `FINAL_T2_GRASP` | `FINAL_T2_BIN` | `FINAL_T2_COLLISIONS` | `FINAL_T2_TIME` | `FINAL_T2_SCORE` |
| 3 | `FINAL_T3_ID` | `FINAL_T3_TAG_SEED` | `FINAL_T3_INPUT` | `FINAL_T3_SUCCESS` | `FINAL_T3_REASON` | `FINAL_T3_COLUMN` | `FINAL_T3_ROW` | `FINAL_T3_GRASP` | `FINAL_T3_BIN` | `FINAL_T3_COLLISIONS` | `FINAL_T3_TIME` | `FINAL_T3_SCORE` |
| 4 | `FINAL_T4_ID` | `FINAL_T4_TAG_SEED` | `FINAL_T4_INPUT` | `FINAL_T4_SUCCESS` | `FINAL_T4_REASON` | `FINAL_T4_COLUMN` | `FINAL_T4_ROW` | `FINAL_T4_GRASP` | `FINAL_T4_BIN` | `FINAL_T4_COLLISIONS` | `FINAL_T4_TIME` | `FINAL_T4_SCORE` |
| 5 | `FINAL_T5_ID` | `FINAL_T5_TAG_SEED` | `FINAL_T5_INPUT` | `FINAL_T5_SUCCESS` | `FINAL_T5_REASON` | `FINAL_T5_COLUMN` | `FINAL_T5_ROW` | `FINAL_T5_GRASP` | `FINAL_T5_BIN` | `FINAL_T5_COLLISIONS` | `FINAL_T5_TIME` | `FINAL_T5_SCORE` |

- [ ] Replace every `FINAL_T*` token from the five frozen source artifacts and recordings.
- [ ] Derive accuracy, grasp success, timing and scores only from these trials.
- [ ] Distinguish observed/manual scores from official evaluator scores.
- [ ] Do not count synthetic tests or interrupted development retries as final trials.
- [ ] Run the source-backed summarizer against exactly the five final summaries:

```bash
python3 src/erc_phase1_solution/scripts/summarize_trials.py \
  results/trial_FINAL_T1_ID_summary.json \
  results/trial_FINAL_T2_ID_summary.json \
  results/trial_FINAL_T3_ID_summary.json \
  results/trial_FINAL_T4_ID_summary.json \
  results/trial_FINAL_T5_ID_summary.json \
  --csv results/final_trials.csv \
  --markdown results/final_trials.md
```

- [ ] Cross-check generated tables against recordings and raw contacts; the summarizer intentionally rejects missing, incomplete, conflicting or more-than-five inputs.

## 8. Report gate

The supplied rules limit the report to five pages excluding the title page.

- [ ] Title page includes `TEAM_NAME`, `UNIVERSITY`, team members and final simulator tag.
- [ ] Architecture explains the four ROS nodes, state machine and competition interfaces.
- [ ] Perception describes live RGB-D use, marker/book/bin detection, depth/deprojection, stability filtering and measured accuracy.
- [ ] Navigation explains the custom odometry-frame controller truthfully; do not describe it as Nav2.
- [ ] Include planned and executed RViz path screenshots from real trials.
- [ ] Manipulation describes URDF-based IK, the left-arm sequence and public gripper topic; do not describe it as MoveIt.
- [ ] Results cover all five trials with detection accuracy, grasp success, elapsed time, contacts/collisions and observed scores.
- [ ] Discuss failures, limitations and future improvements.
- [ ] Label all figures and ensure every quantitative claim can be traced to a saved artifact.
- [ ] Export to PDF and confirm the page count and readability at normal zoom.

Suggested five-page body allocation:

1. System architecture and mission state machine.
2. Perception method and quantitative detection results.
3. Navigation method, timing, and planned/executed path evidence.
4. Manipulation method and grasp/place evidence.
5. Five-trial results, limitations and future work.

## 9. Code and artifact packaging

- [ ] Include `src/erc_phase1_solution` with `package.xml`, `setup.py`, package README, launch, config, assets and source.
- [ ] Confirm `config/solution.rviz` is installed and opens when `rviz:=true`.
- [ ] Remove caches and generated bytecode from the submission archive (`.pytest_cache`, `__pycache__`, `*.pyc`).
- [ ] Exclude unrelated vendored build/install/log directories unless the organizer requests the full simulator repository.
- [ ] Include only representative evidence/log files if the submission form permits them; retain the complete originals privately.
- [ ] Test extraction and build from a clean directory on the frozen official environment.
- [ ] Check that no absolute host paths appear in launch/config/source files.
- [ ] Record the final Git commit SHA and save checksums for the report PDF and original video.

## 10. Final upload

- [ ] Confirm the organizer's current form URL, deadline timezone, allowed formats and size limits.
- [ ] Submit the public team GitHub repository link at the frozen final commit.
- [ ] Submit the YouTube link for the unedited video, maximum five minutes.
- [ ] Upload the report PDF, maximum five body pages excluding the title page.
- [ ] Verify the GitHub and YouTube links are accessible and the uploaded report opens and matches its checksum.
- [ ] Save the submission confirmation, timestamp and any receipt/confirmation email.
- [ ] Keep the frozen repository, container/image identifier, original video, artifacts and trial logs until results are final.

Final sign-off:

| Role | Name | Date | Signature/approval |
|---|---|---|---|
| Technical lead |  |  |  |
| Report reviewer |  |  |  |
| Submission owner |  |  |  |
