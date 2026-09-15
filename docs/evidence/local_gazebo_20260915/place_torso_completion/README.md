# Registered PLACE torso completion

Trial `8d722ba0fcbf` showed that the 2.2 s torso action could report success while
the body remained 12 cm below its target and was still rising at 0.035 m/s.
The following arm movement therefore started before the torso reached the
height assumed by its checked route.

## Change

The accepted registered PLACE plan supplies the same fixed-arm torso path and
endpoint. Immediately before sending the goal, fresh measured joint values and
the deployed URDF velocity limit determine its serialized duration:

`max(2.2 s, abs(target - measured_start) / (0.8 * velocity_limit))`

A rise from 0.10 to 0.35 m uses **8,928,571,429 ns**. The ordinary action watchdog
uses that same duration. Existing contact, scene, payload and cancellation
monitoring remain active. A valid existing completed-hold shortcut still avoids
a redundant command.

After action completion, a separate measurement collector requires a fresh
100 ms stationary span at `torso_ready`, before any PLACE arm leg. It reuses the
existing closed-pose predicate: 1 mm torso error, 2 mrad arm error, 1 mm/s torso
speed and 1 mrad/s arm speed, plus the original gripper and odometry limits.
The 0.5 s simulation settling limit, 3 s producer-progress watchdog, 150 ms
freshness and 75 ms maximum gaps are unchanged. A failed arrival raises before
recovery can assume the unachieved endpoint. The hand stays closed.

The collector closes before the ordinary arm handoff check. Four new journal
events record admitted duration, actual start, target, velocity limit, result
and at most 12 raw samples. Official controller code and settings are unchanged.
The positive first-arm movement retains its restored 0.8 s duration.

## Verification and provenance

The new helper passed **18 tests in 0.71 s**, using actual runtime sender/caller
methods, controlled joint feedback and an exact archived registered scene.
They cover feasible timing, premature controller success, measured settling,
completed-hold preservation, identity/freshness failures and collector cleanup.
This is software evidence, not a Gazebo success claim.

Production checks covered **411 distinct cases across overlapping runs**. The
initial run passed 389 with three old fixture failures. Those fixtures were
scoped to their intended behavior, and all affected checks passed on rerun.
Additional journal checks passed. The counts must not be added; this is not a
new full-suite run. Logs and `source_manifest.json` preserve the exact scopes,
reviewed runtime bytes and test-only fixture changes. Physical validation of
this correction remains pending.
