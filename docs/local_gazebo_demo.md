# Local Gazebo demonstration

## Verified result — 15 September 2026

Clean trial `9c768db2db9e` completed the seed-101, shelf-2, red-book mission
in **318.006 seconds (5 minutes 18 seconds) from solution launch**.
Simulator startup is additional. All six navigation goals succeeded, and the
independent simultaneous-pose check put all eight book corners inside the
conservative bin cavity, with 66.65 mm minimum side clearance.
The robot finished at the release pose; this profile does not return the hand.

The earlier corrected diagnostic trial `d5792498160e` also delivered the book,
in 484.229 seconds from solution launch. That run included compilation and
viewer changes, so it is not a controlled speed comparison.

The changed settling modules passed **311 focused tests**. Final evidence:
[run review](evidence/local_gazebo_20260915/reference_column2_red/run_review.json),
[mission summary](evidence/local_gazebo_20260915/reference_column2_red/trial_9c768db2db9e_summary.json),
[independent containment](evidence/local_gazebo_20260915/reference_column2_red/containment.json), and
[screenshot](evidence/local_gazebo_20260915/reference_column2_red/completed.png).

## Startup correction

The first local trial (`c008dbd46c34`) stopped with
`initial stow measured stop wall deadline`. Gazebo was running at roughly
0.13 times real time. The three-second wall limit cut short a settling interval
measured in simulation time, after the torso controller had reported success.

The corrected stow wait uses simulation time for settling. A separate wall
watchdog still detects a clock or sensor stream that has stopped advancing,
and the command timeout bounds the entire wait. Measured positions, velocities,
freshness, cancellation and ownership checks still apply.

The same clock-domain issue was corrected in the empty-torso planning wait,
placement transition stop and the pre-release clock catch-up. Regression tests
exercise slow, frozen and invalid feedback, as well as the original behavior
when simulation time is inactive.

## Viewer and launch

The spectator window runs separately from the simulator, using Ogre1 and a
small GUI configuration. A local GUI plugin limits the spectator to 15 frames
per second and adjusts its lighting so white surfaces remain visible.
The simulator continues to render the robot's sensors
with Ogre2. These viewer changes do not alter the world, robot models, physics,
sensor images or control limits.

The launcher also selects an optimized Release build of the unchanged
`gz_ros2_control` source. This avoids the unoptimized C++ build produced by
the original plain `colcon build`. Its separate `build_release/` and
`install_release/` directories preserve the earlier build for comparison.

Inside the prepared container, after closing any previous simulation:

```bash
cd /opt/erc_ws
/entrypoint.sh tools/run_gazebo_demo.sh
```

From this computer's host, using the prepared `erc_friend_demo` container:

```bash
powerprofilesctl launch --profile=performance --reason='ERC Gazebo mission' \
  --appid=erc-demo docker exec -it erc_friend_demo \
  /entrypoint.sh /opt/erc_ws/tools/run_gazebo_demo.sh
```

The temporary performance profile is released when the command exits.
The launcher checks camera, laser, joint-state, odometry and clock messages
before starting one mission. Defaults are seed 101, shelf column 2 and red.
`ERC_SEED`, `ERC_SHELF_COLUMN` and `ERC_BOOK_COLOUR` override these values.
Export them inside the container, or forward host values with Docker options
such as `docker exec -e ERC_SHELF_COLUMN=2 -e ERC_BOOK_COLOUR=red ...`.
Logs go to a new timestamped directory under `results/`.

The window stays open after a mission result. Close it or press Ctrl+C in the
launch terminal to stop that run. A visible window is not proof of completion;
check `trial_*_summary.json` and verify the delivered book inside the bin.

## Column 4, blue trial — 15 September 2026

The visible trial `d0fcc30b05eb` reached the selected shelf and book, then
stopped before grasping with `book_reacquisition_failed`. The blue book
remained visible, but its close-range depth surface failed the unchanged
metric tracking gate. No grasp was attempted and no collision episode was
reported. This target is **not verified to complete**.

[Trial evidence and camera diagnosis](evidence/local_gazebo_20260915/column4_blue/README.md).

Use `./tools/show_gazebo.sh 4 blue` to reproduce this target, or
`./tools/show_gazebo.sh 2 red` for the previously successful local baseline.
The demonstrated column 4 run used the same 453 solution-source files as the
restored column 2/red baseline.
