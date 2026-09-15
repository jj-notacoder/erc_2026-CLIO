# Lower-shelf planning: offline parallel comparisons

Two independent A/B comparisons replayed the onboard inputs of column-4/blue trial `121f7ff7e8ad` through the complete lower-shelf planner. All four runs returned the **same complete plan**, including approach, lift, carried return, cached compaction, and recovery. They issued **no controller commands**.

## Observed timings

Seconds are host wall time inside the offline helper. Each mode used a separate fresh Python process and fresh geometry/cache objects. **“Cold” refers to those process caches; OS/filesystem caches were not flushed.** These are single observations, not repeated averages.

| Comparison | Measured scope | Before | After |
| --- | --- | ---: | ---: |
| Empty geometry pool | Helper entry to lift-stage boundary | 25.532 s | 18.274 s |
| Empty geometry pool | Complete lower helper | 46.697 s | 40.248 s |
| Loaded continuation pool | Carried-return stage | 17.714 s | 15.107 s |
| Loaded continuation pool | Complete lower helper | 39.020 s | 36.340 s |

The second pair already uses parallel empty geometry in both modes. Its “serial” label applies to the carried-return continuation; the preceding lift still uses its existing worker pool. Compare rows within each pair: these separate runs are not a single live end-to-end timing sequence.

- [Empty comparison](empty_comparison.json): only `lower_shelf_pick.py` changes between package snapshots. The new empty pool closes before loaded planning starts.
- [Loaded comparison](carry_comparison.json): both modes import the same package snapshot; only `pickup_post_retreat_parallel_geometry_enabled` differs. The enabled mode retains the loaded pool through the carried-return continuation.
- Both empty checks retain **613 checked samples, 264 cache hits, and the same exact cache digest**. Each real pool created and reaped eight workers. The loaded continuation consumed 4,067 queries, with none discarded.

The pool's `consumed` count includes ordered cache lookups and differs from the checker's unique `samples` count; the comparison JSON preserves both scopes.

## Exact input and output

[Onboard inputs](onboard_inputs.json) preserves the recorded joint context, aperture, target, odometry pose, independent marker/bay context, and comparison approach. It uses no Gazebo world-model poses.

- Front target in `base_footprint`: `[0.650858022288654, -0.05180938644780962, 0.927929015120235]` m.
- Recorded context at ROS 36.152 s: torso approximately 0.10 m; head approximately `[0, -0.4]`; parked right arm approximately `[-0.36, -1.83, -0.47, -2.35, 0, -1.2, 0]` rad. The JSON retains full precision.
- Aperture: `0.06899999999927978` m from joint feedback at ROS 36.150 s. The original timestamps remain provenance; the offline harness supplies a fixed replay clock and fresh fixture stamps.
- The helper selects the row-3 positive-wrist candidate with torso 0.10 m, pitch +0.25 rad, roll π, 25 mm grasp-depth offset, 0.12 m carry extension, +0.18 m staging raise, and `compact_path='middle'`.

[Common plan](common_plan.json) contains the identical complete result once: six approach solutions, one setup waypoint, two extraction targets, three lift/withdrawal legs, 31 cached compaction legs, and seven unloaded recovery waypoints. The planned approach differs from the recorded approach by exactly zero. Fields named `actual_grasp_*` in the input are planned-joint FK, not physical achieved-pose measurements.

[Result summaries](result_summaries.json) retain each run's pass status, source/model identity, empty-cache state, timings, plan digest, and controller-call list. Full-result hashes are recorded in [provenance](provenance.json); repeated plan arrays, worker PID details, and full logs are omitted.

## Replay method

The original benchmark scripts are identified by hash in provenance. Their method was:

1. Use the ROS 2 Humble repository environment, official description packages and `_official_manipulation_planner` test fixture. Construct a new node and immutable official mesh objects for each run, with eight geometry workers and numerical-library threads limited to one.
2. Install the full-precision recorded start/head/right/aperture into the fixture. Set its clock, joint stamps, and odometry stamp to 1 s; keep odometry pose from the input and speeds zero. This admits replay inputs and **does not validate live freshness or drift**.
3. Register `RelativeShelfBay` by applying inverse odometry yaw to the recorded marker displacement and inward normal. Use lateral uncertainty 0.200 m, roof uncertainty 0.010 m, and the explicit upright-supported-book assumption. Use the production registered shelf/body/tool predicates unchanged.
4. Capture `EmptyPickupCollision`. Call `plan_lower_shelf_pick` with the front and bay. Its lift callback calls `run_pickup_geometry(plan_lift_first_extraction, ...)` with aperture 0.020 m, lift 0.020 m and modeled tool allowance 0.005 m; the loaded comparison forwards the optional carried-return continuation.
5. Replace actuator entry points with assertions. Keep actual installed source/model identity verification and real subprocess ownership. Record the complete returned plan, cache state, exact approach difference, stage timings, and worker cleanup.
6. Run the modes sequentially in separate processes, using the package snapshots and feature choices above. Compare complete JSON plan objects and empty-cache state exactly, without numeric tolerance.

The common plan's canonical SHA-256 is `09d52a587235c6086f99c10dc832e3e8a3cf7ee20651bdfee12c67ba4b33b0fc`. The digest encoding is UTF-8 `json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)`. It matches all four retained result summaries.

## Source scope and limits

[Empty source manifests](empty_source_manifests.json) preserve common modules and the two helper hashes. [Loaded benchmark manifest](carry_source_manifest_benchmark.json) pins the source that produced the timings. The [final reviewed manifest](carry_source_manifest_reviewed.json) and [changed-file hashes](carry_changed_files_reviewed.json) additionally record a later exception-only correction in `pickup_geometry_backend.py`: preserve an original latched backend failure even when a caller translates it into a no-route exception. The author ran focused exception-path tests after that correction; the positive A/B was not rerun. The timings remain attributed to the benchmark snapshot.

These offline results do not establish live planning latency, sensor freshness, physical grip retention, delivery, performance for every book, or completion under five minutes. The source trial itself [acquired the correct book and stopped before lifting](../column4_blue_grasp/README.md). No simulator, benchmark, IK/mesh computation, or test was rerun while assembling this evidence directory.
