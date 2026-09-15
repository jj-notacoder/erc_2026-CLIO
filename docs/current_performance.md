# Current performance — 14 September 2026

This release contains the exact **453-file candidate69 / revision117** package used by Run97 and Run99. The installed `collision_quality` profile is selected by the normal public launch. It uses the official position controller with a **0.0182 m** gripper master target, an actuator coordinate rather than measured pad separation. Official models, controllers and physics are unchanged.

| Measurement | Run97 | Run99 |
| --- | ---: | ---: |
| Solution launch to completion | **4:36.906** | **4:43.676** |
| Simulator start marker to completion | **approximately 4:52.546** | **approximately 4:58.012** |
| Strict final containment | Passed | Passed |
| Stable samples / simulated span | 22 / 23.006 s | 22 / 20.992 s |
| Minimum final side clearance | 66.213 mm | 0.780 mm |

Both completed trials used the same headless seed-101 scenario, requested shelf column 2 and a red book, on the development computer. The startup marker has one-second precision. Build time and the extra observation period after completion are excluded. GUI timing, other target books, broader reliability and a complete five-minute competition video remain unverified.

Run98 failed its sensor-readiness check before mission dispatch between these two completed trials. Both laser streams and the depth camera image/info were absent. Its failure was preserved and its world closed before the fresh Run99 startup; it is not counted as a completed mission.

## Physical result and completion boundary

Both completed trials picked the requested book, carried it to the bin and passed independent final-containment analysis. Mission completion occurs at the checked stationary release with confirmed target/bin contact. **Final hand return was not performed.** The additional containment observation follows completion and is excluded from the timing above.

Placement remains a **drop**: the last measured book bottom before opening was approximately **267 mm** above the bin floor in Run97 and **262 mm** in Run99. The different final side margins show landing variation. These observations do not establish gentle handling or continuous collision clearance.

## Run this version

Build and source this checkout in the official ROS workspace:

```bash
cd /opt/erc_ws
colcon build --symlink-install
source install/setup.bash
ERC_SEED=101 ROS_DOMAIN_ID=31 ros2 launch erc_bringup simulation.launch.py \
  headless:=true depth_cloud:=false
```

In a second sourced terminal:

```bash
ROS_DOMAIN_ID=31 ros2 launch erc_phase1_solution solution.launch.py \
  shelf_column_number:=2 book_colour:=red team_name:=TEAM_NAME
```

Replace the team name and assigned target. These commands select the tested task and headless world; the reported trials also used sensor-readiness checks and independent observation. The launcher loads the installed `solution.yaml` and then `collision_quality.yaml`; all 63 selected values match the tested profile. No separate candidate launch file is required. Its 1,800-second development watchdog is separate from the five-minute video limit. `profile:=default` selects older base settings. Rebuild and source the install after updating.

Selected changes include shorter initial serial arm positioning, checked withdrawal and placement timing, a loaded PLACE speed cap of 2.0, and a separately guarded release-only connector. Normal translation and yaw gains are 1.8 on the selected normal-navigation path. Existing speed caps, acceleration, obstacle, freshness, stopping, retention, collision and measured-endpoint checks remain active. The changes were exercised together; individual time savings are not isolated measurements.

## Software and publication evidence

The current navigation change passed **534 affected tests and three fresh constructor modes** on candidate69. Its direct parent supplies the separate **659-test** connector-fix result. The earlier **2,263-test** affected result supplies the navigation-module evidence used by the current focus; the **5,986-test Full63** result is an ancestor. These counts describe distinct snapshots and are not added together or presented as a new full-suite run. All 453 current package files, 895 official files and 40 resolved assets were checked. The exact package was built and executed for both completed trials.

The [compact evidence bundle](evidence/five_minute/README.md) records the current profile, software evidence, results and timing/containment summaries. [Testing instructions](testing.md) retain the portable source fixtures and required ROS environment. Raw recordings, build products and unselected prototypes are not needed to run this release and are excluded from the compact bundle.

## Historical Run76 release

The previous candidate34 release contained 423 package files and used a 0.0181 m gripper target. Run76 completed its routine, including measured hand return, in **5:47.536** from solution launch; its later GUI demonstration took **6:26.719**. Startup was additional. Run76's strict final containment passed with a 0.344 mm side margin, and placement involved a drop from approximately 262 mm above the floor. Those results use a different completion boundary from the current release.

The [Run76 evidence](evidence/run76/README.md) retains its 84-test validation49 result and separately inherited 5,451-test Full48 certificate. Run74's strict-containment failure and Run75's pre-pickup head-timing failure remain historical development outcomes. None of these older results is relabeled as current candidate69 evidence.
