# Phase 1 Gazebo WIP checkpoint — 2026-09-05

This branch is a resumable engineering checkpoint. It is **not** evidence of a
completed Phase 1 mission and should not be presented as a successful official
trial.

## Saved progress

- Production work in `erc_phase1_solution` adds stricter target identity,
  payload-retention, collision, recovery, timeout, and carried-navigation
  checks, together with focused tests.
- Retry 25 (`7c530c85f052`) made the first repeatable physical top-row grasp:
  adaptive closure acquired the exact red target at 0.030009 m, engaged a
  0.029009 m transport lock, measured fresh bilateral forces of 2.44 N and
  4.04 N, retained the book through all five slow extraction legs, and passed
  the final settle/contact probe.  The mission then lost contact after about
  56-60 mm of the 0.35 m carried base retreat and the book fell to the floor.
- Retry 26 (`12db82e07f93`) tested an 18 mm top-row clearance-target lift.  IK
  produced about 9.6 mm of actual endpoint rise; adaptive closure again
  acquired the exact target at 0.030007 m and measured 2.45 N / 3.36 N before
  all five retained extraction legs and the unsupported settle/contact probe
  passed.  It still lost the book after only 25-27 mm of base retreat.  Gazebo
  reported the final book pose on the floor at `z=0.015 m`, so this was a real
  physical drop rather than contact-sensor dropout.
- These runs rule out perception, IK reachability, initial pressure acquisition,
  and simple shelf-lip unloading as the current blocker.  A vertical fingertip
  pinch is not a robust carried support once the mobile base starts moving.
- Live same-world diagnostics established that a closed-gripper downward slide
  is invalid: the first 9.75 mm hand descent dragged the book about 4.9 mm and
  pitched it about 4.3 degrees at the shelf edge.
- The book was recovered, pressed back onto the shelf, released in stages, and
  left stable. The open fingertip hook acquired both sides, but failed to lift
  because the low-effort fingertip linkage deflected instead of unloading the
  shelf.
- Exact mesh checks found the next single-arm option: use the rigid
  `gripper_left_base_link` cap beneath the exposed book bottom, then close the
  fingers laterally as a cage. The route and guard experiments are retained in
  the `live_rigid_palm_*` scripts. It now needs a guarded production support
  transfer before base motion; the diagnostic scripts use Gazebo truth and
  temporary sensors and therefore cannot be copied into the competition path
  unchanged.

## Next implementation boundary

Do not spend another full run on a larger preload, a slower base profile, a
weaker contact watchdog, or the existing post-retreat compact route.  Previous
preload and slow extraction experiments failed at the same support boundary,
and the cached compact route was collision-checked only after the base has
cleared the shelf.

The next production change is a staged rigid-palm/lower-jaw support transfer
while the robot is stationary at the shelf:

1. Preserve the verified bilateral fingertip pinch and move only through a
   complete shelf-bay and robot swept-volume check.
2. Bring the rigid gripper cap/lower jaw under the exposed book bottom without
   using Gazebo model pose or temporary contact topics.
3. Dwell and require fresh bilateral retention plus bounded controller error;
   fail closed while the book is still over the shelf.
4. Permit the base retreat only after the book has mechanical support against
   gravity, then run the already cached post-retreat compact route.

## Verification status

- Earlier host-side suite: 118 tests reported passing.
- The two focused top-row lift/pick planner checks passed, and
  `erc_phase1_solution` built successfully with `colcon`.
- Per request, the complete suite was not rerun.  Retry 26 was a complete
  seed-101 mission attempt and ended `success=false` at the carried retreat.
- The official randomized multi-trial evidence is still outstanding.

The files named `live_*`, `search_*`, `scan_*`, `validate_*`, and related probe
scripts are temporary engineering diagnostics. Keep them while resuming this
checkpoint; remove or relocate them before a final competition submission.
