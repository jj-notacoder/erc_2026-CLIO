# Collision-quality development profile

Select `profile:=collision_quality` for this explicit development path. The
ordinary column/colour-only launch retains legacy defaults. The current
[revision57 record](revision57_validation.md) reports **3,706 full ROS tests**,
selected/default four-node constructors and unchanged official assets.
**Run50 completed pickup, delivery and measured arm/torso HOME in 14 min 15.558 s
wall time with the Gazebo GUI closed.** The whole book settled inside the bin,
with 2.537 mm minimum side clearance. The final release was a drop onto the bin
floor; gentle placement remains unfinished. Sampled geometry found no queried
arm/tool overlap, with three unavailable measured tool links and gaps between
samples. This is one verified run, not continuous collision proof or a
competition-ready default launch. The [portable evidence](evidence/revision57/README.md)
preserves the validation and physical outcome separately.

Build and source the team package in the official ROS 2 Humble workspace:

```bash
cd /opt/erc_ws
colcon build --symlink-install --packages-select erc_phase1_solution
source install/setup.bash
```

Start the unchanged official simulator in one terminal:

```bash
ERC_SEED=101 ros2 launch erc_bringup simulation.launch.py headless:=false depth_cloud:=false
```

After sensor readiness, source the same workspace in another terminal and run:

```bash
ros2 launch erc_phase1_solution solution.launch.py \
  shelf_column_number:=2 book_colour:=red profile:=collision_quality \
  trial_timeout_seconds:=5400 team_name:=TEAM_NAME
```

Use the assigned column and colour. Seed 101 is the development comparison
scenario, not varied-trial evidence. The installed [profile YAML](../src/erc_phase1_solution/config/collision_quality.yaml)
is layered after base YAML; explicit launch arguments retain precedence. The
5400-second mission allowance and selected 1800-second planning allowance are
development settings. They do not satisfy the five-minute unedited video limit.
The ordinary launch timeout remains 270 seconds; default planning is 420 seconds.

| Selected setting | Value |
|---|---|
| Public stock gripper target / initial lift | .017 m / 5 mm |
| Placement transport arm speed scale / pre-open stationarity | 3.0 / enabled |
| Placement torso height | .35 m |
| Book-center clearance / release above fitted bin floor center | .50 m / .270 m |
| Initial placement TCP clearance X | .60 m |
| Bin navigation standoff | .82 m |
| Manipulation command / mission manipulation wait | 2400 s |
| Selected search / explicit mission allowance | 1800 s / 5400 s |
| Development camera / observer cap | 5700 s |

The bin's known shape is fitted to onboard RGB-D to estimate its floor-center
frame and orientation. Placement uses that frame for both the book-centered
target and bin collision material, together with the table fit from the same
depth epoch. Invalid, stale or mismatched scene evidence prevents placement.
The legacy front-patch `PointStamped` used for bin navigation is preserved.
Evaluator/world poses do not supply controller targets or scene geometry.

Pickup setup and the selected placement path check existing robot meshes, a
conservative head-screen visual envelope and the nine nominal gripper surfaces.
Placement also checks the held book, finite table solids and hollow bin material,
with modeled and measured registration margins. It retains signed support,
joint limits, every loaded segment, full opening and the entire open-hand return.
Measured torso/arm endpoints are awaited before the next checked command, with
fresh parked-arm/head/gripper context and cancellation. Observed arm/tool contact
with the bin or table latches a stop through release and return. Current
revision42 also serializes ordinary retained PLACE publication with that stop
lock and rechecks cancellation/watchdog state before sending. Per-sample bounds
reuse preserves the original current world arrays and exact reductions; it adds
no cross-pose verdict cache or relaxed collision test. One ordered Run38-input
replay pair took 191.582 versus 181.943 seconds with identical paths and all
2,337 sample records. Its 5.03% reduction is an instrumented fixed-context result,
not a live speedup or a change to the profile. See the [scope and provenance](revision42_validation.md#bounded-nominal-replay).

For fitted-bin placement, the planner tries ten proximal setup steps before the
twenty-step fallback. The accepted recorded-input replay expanded to **12 setup
legs**, compared with **22** in the earlier, different nominal fixture; this is
not a controlled motion-speed comparison. Every nonzero scene leg retains at
least 61 samples. The search now tries to finish a tight Cartesian candidate
before expanding the original beam, under one shared configured deadline and unchanged global limits
of 512 IK calls, 24 paths and six full candidates. No saved joint route is used.

Contact aggregation computes each delivered entry's force once and updates the
latest history incrementally, retaining the original out-of-order fallback and
all fault/identity checks. An equal-input archived-gripper replay used **46.7%
less callback CPU**; this is not a measured planner or mission speedup. Executor
configuration and official physics/controllers remain unchanged. The selected
four team processes retain single-thread BLAS/OpenMP settings.

The public stock command bypasses the experimental force/effort-amplitude and
pressure-qualification policies, while preserving official actuator limits and
named-contact monitoring. Master position is not measured jaw separation.
Nominal geometry does not prove passive-finger shape, a rigid book attachment or
release dynamics; Run40 again exceeded its original nominal attachment during
the long stationary hold. See the [completed physical record](revision42_validation.md#completed-run40-physical-result).

Completion requires measured opening, checked hand return and fresh
attempt-correlated bin contact. It reports `placement_operation_completed` and
`released_with_bin_contact`; `physical_inside_verified` stays unknown without
positive onboard proof. Independent physical containment, unintended contacts
and elapsed time must be reported separately. Historical Run33/34 deliveries
had collisions; they do not validate the revised route.
