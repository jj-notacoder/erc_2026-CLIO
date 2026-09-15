# ERC Phase 1 Solution Package

`erc_phase1_solution` is the evaluator-facing ROS 2 Humble package for the
Emirates Robotics Competition 2026 simulation phase. It targets the official
`erc_sim_2026` main commit `b1f9b05e20f4750b59f3321d88f172cc4dbb1386`
and uses one TIAGo Pro arm. The public interfaces are unchanged from v1.0.3;
the organizer's newer checked-in book and robot assets are now adopted.

## Current baseline and status — 11 September 2026

The [official 9 September notice](https://github.com/dfl-rlab/erc_sim_2026/blob/b1f9b05e20f4750b59f3321d88f172cc4dbb1386/README.md)
specifies 20 mm books. The maintainer explicitly merged these updates into
main for teams to use. The public interface baseline is v1.0.3.
The repository's [environment record](../../official_environment.json) and
[official-reference guide](../../tools/official_reference/README.md) pin the
source and explain verification. Use the published checked-in URDF; the older
generator does not reproduce its new gripper values.

Current payload dimensions are `[0.16, 0.02, 0.25]` m (depth, thickness,
height). The preclose command is 0.025 m, and the minimum accepted primary
joint position is 0.0155 m with a 0.0005 m margin. Joint coordinates are not
jaw separation. The default adaptive position-only path uses fresh bilateral target contact,
bounded preload and force/effort/width checks. The explicit collision-quality
profile instead uses the ordinary stock 0.017 m command with named-contact
retention monitoring; setting a target alone is not proof of retained cargo.

Run39 completed physical pickup, transport, release and measured arm return
using unchanged official assets and the collision-quality development profile.
The book settled wholly inside the bin and stayed there for the final 31
simulated seconds; the bin did not move. No arm/tool contact with the table or
bin was recorded, and retrospective sampled screen/table/bin checks found no
overlap. The nearest final side clearance was only 0.475 mm. Long stationary
planning also allowed about 17 mm / 21 degrees of book-to-hand drift, exceeding
the original nominal attachment envelope. These limitations remain relevant
despite the successful delivery. Seed 101, column 2 and red are the demonstrated
case; this is not evidence across other targets or competition timing. See the
[current validation record](../../docs/official_current_validation.md).

These historical runs used an external development launcher and parameter
overrides, including a 0.017 m stock gripper target. The public launch below
now selects the installed collision-quality configuration by default. This
launch-default correction passed its focused launch checks and awaits physical
qualification; the historical runs do not validate the new default command.
The explicit profile:=default option retains the base solution.yaml settings. Older v1.0.3
results are historical evidence; Runs33 through39 concern the updated 20 mm model.

## Launch

Start the official simulator first:

```bash
ros2 launch erc_bringup simulation.launch.py
```

Then launch the solution from the repository root:

```bash
ros2 launch erc_phase1_solution solution.launch.py \
  shelf_column_number:=2 book_colour:=red
```

Both arguments are required. Valid columns are `1` through `5`; valid colours
are `red`, `blue`, `green`, and `yellow`. Optional development arguments are
`team_name`, `rviz`, `dry_run`, `trial_timeout_seconds`, and `profile`.

The two required arguments select collision_quality and the 1,800-second
development watchdog. The equivalent explicit command is:

```bash
ros2 launch erc_phase1_solution solution.launch.py \
  shelf_column_number:=2 book_colour:=red profile:=collision_quality \
  trial_timeout_seconds:=1800
```

To select the base configuration explicitly, add `profile:=default`. Add
`trial_timeout_seconds:=270` as well to restore the previous public timeout.
Node constructor defaults and both YAML files are unchanged. The selected profile
uses a .35 m torso, .50 m clearance height, .270 m book-center release height,
.60 m approach and mission-owned .82 m bin standoff. Onboard RGB-D supplies
the bin floor-center and orientation together with the table geometry from
the same depth epoch. Measured motion endpoints, opening and return evidence
remain required. Only its four team processes receive single-thread
BLAS/OpenMP settings.

Revision41 passed 2,668 tests and actual selected/default four-node constructor
checks. Run39 used that exact source. Its nominal full-route search completed
in 1,229.878 wall seconds within the selected 1,800-second allowance, then
the physical placement and return completed. Earlier Run38 passed its nominal
geometry checks but exceeded its 900-second search allowance before motion.
The 1,800-second mission watchdog above is a development allowance, separate
from planner and command timeouts. It does not meet or redefine the five-minute
performance/video target. Record the measured elapsed time and report a missed
target honestly; a longer watchdog is not evidence of timely completion.

Two subsequent controller changes are recorded separately from Run39's physical
evidence: ordinary retained PLACE sends share the contact-stop command lock
and recheck cancellation/watchdog state before publication; scene checks reuse
lazy bounds only for private world arrays within one sample. Collision
predicates, sampling, path selection, grasp settings and official physics are
unchanged. Source-specific software validation and later physical trials are
reported in the linked validation record; Run39 does not certify subsequent
source bytes.

The follow-up records major planning stages and exact onboard planning inputs.
For parked body pairs, a sufficient separation test projects every current
world-space mesh vertex, including all components, onto candidate directions.
A clear result requires a gap greater than 100 micrometres plus numerical
allowance. Cached directions are hints only; every use checks the current
geometry again. Uncertain, touching or overlapping projections retain the
detailed triangle and containment test. The separation proof can remove
legacy numerical false positives on degenerate facets, so it does not claim
universal equivalence to the old Boolean result. It does not change official
geometry or physics, or certify physical model accuracy. Software checks and
fresh physical results are reported separately in the
[profile guide](../../docs/collision_quality_profile.md) and
[validation record](../../docs/official_current_validation.md).

## Nodes

- `perception_node`: live RGB-D marker, book, and bin detection; timestamped
  evidence capture. Overview row IDs are withheld until all four physical
  shelf rows are visible, preventing partial views from shifting row numbers.
- `navigation_node`: conservative odometry-frame holonomic waypoint control
  using both official laser topics.
- `manipulation_node`: URDF-derived left-arm/torso IK plus the official public
  gripper command topic. Carried-return planning checks the inflated book
  against official base, torso, head/camera and both-arm collision meshes
  before closing the gripper, with exact nonadjacent-link self-collision and
  closed-mesh containment checks at endpoints and adaptive interior samples;
  parked-arm geometry comes from measured joints. An unsafe HOME fold uses a guarded
  compact upright transport posture, and bounded roll-preserving waypoints
  avoid inverting it on the way to the bin. Free-joint IK reserves 0.01 rad
  from URDF hard stops and placement reserves 0.03 rad. If no compact
  payload-safe route exists, the pick is rejected before motion.
- `mission_manager`: bounded mission sequencing, required scoring publications,
  contact verification, collision telemetry, and JSONL trial logging.

### Historical v1.0.3 transport geometry

The following retry-19 figures describe an older v1.0.3 configuration. They
are preserved as regression evidence, not current route certificates. That
configuration first backed the extended lowered payload
0.70 m straight away from the shelf without changing heading. It then requires
a compact fold before any turn: the retry-19 modeled base/torso/head/arm-link/book
envelope is base-limited to a 0.436 m planar radius, its book radius is 0.418 m,
and the 10-leg checked fold peaks at 42.43 degrees of tilt. An earlier 601-sample
exact triangle-surface audit passed; the containment-enabled route also passed
all native checks and a 100-sample dense spot audit. After the 0.70 m retreat, the
inflated payload retains 0.519 m beyond the configured shelf margin throughout
the fold. The full carried navigation path
still requires live arena validation against the effective controller
clearance.

That historical generalized collision model ended at arm links 1-7. The
retry-19 route received a separate palm-mesh audit. Later working-tree changes
added gripper/tool checks, but historical certificates do not generalize to
other rows or to the updated official robot/book assets.

The package intentionally does not claim Nav2 or MoveIt integration. The
official source includes them without a competition-specific navigation or
manipulation configuration used by this package.

## Current official interfaces used

- RGB: `/head_front_camera/head_front_camera/color/image_raw`
- Depth: `/head_front_camera/head_front_camera/depth/image_rect_raw`
- Depth intrinsics: `/head_front_camera/head_front_camera/depth/camera_info`
- Odometry and base command: `/odom`, `/cmd_vel`
- Lasers: `/scan_front_raw`, `/scan_rear_raw`
- Arm action: `/arm_left_controller/follow_joint_trajectory`
- Head action: `/head_controller/follow_joint_trajectory`
- Torso action: `/torso_controller/follow_joint_trajectory`
- Public gripper topic: `/gripper_left_controller/joint_trajectory`
- Contacts: `/contacts`, `/bin_contacts`
- Required score topics: `/erc/shelf_column_identification` and
  `/erc/shelf_row_identification`

In the pinned current source, the public gripper topic is relayed through the official clamp node
to `gripper_left_controller_raw`. This package never bypasses that clamp.
The default adaptive pick controller closes that position-only interface in 1 mm steps. It
accepts a grasp only after three fresh, synchronized force samples from both
target-book fingers, then applies one relative 1 mm preload and requires a new
bilateral force window. Palm-only, unilateral, stale, over-force, moving-joint,
and implausible-width states stop the close and enter the existing recovery.
Joint effort is used only as an overload guard, never as proof of contact.

## Outputs

Annotated live images are written atomically to the repository-level
`erc_images/` directory. Development JSONL and summary records go to
`results/`. Each completed navigation goal also creates a report-ready JSON
artifact under `results/paths/` containing timestamped planned and executed
poses, path lengths, purpose, and outcome; the trial summary indexes those
artifacts. `ERC_ERC_IMAGES_DIR` and `ERC_RESULTS_DIR` can override these paths.

## Validation

Pure-Python tests can run without ROS installed:

```bash
python3 -m pytest test -q
```

ROS build and launch validation must be performed inside the official Docker
environment:

```bash
colcon build --symlink-install --packages-up-to erc_phase1_solution
colcon test --packages-select erc_phase1_solution
```

The current official gripper still accepts position commands, not a custom
effort command interface. Historical v1.0.3 failures do not establish the
updated model's outcome; a successful trajectory command alone is not proof
that a book stayed grasped.
The bilateral evidence above comes from the simulator's `/contacts` wrenches,
not from physical fingertip pressure sensors.
