# Pinned competition source — 11 September 2026

Use untouched official main commit
`b1f9b05e20f4750b59f3321d88f172cc4dbb1386` from
[`dfl-rlab/erc_sim_2026`](https://github.com/dfl-rlab/erc_sim_2026/tree/b1f9b05e20f4750b59f3321d88f172cc4dbb1386).
This adopts a published organizer update; it does not assume permission for
team changes to physics, robot assets or controller interfaces.

## Why this reference

The original Phase 1 brief names v1.0.0. The latest numbered release remains
[v1.0.3](https://github.com/dfl-rlab/erc_sim_2026/releases/tag/v1.0.3), at
`0a09806ecbade5edc9f8a148b7c9f439ed761554`, published on 24 August.
The organizer subsequently [merged PR 6](https://github.com/dfl-rlab/erc_sim_2026/pull/6)
on 8 September and [announced main was ready for teams to continue development](https://github.com/dfl-rlab/erc_sim_2026/issues/2#issuecomment-5582821341).
The [9 September README notice](https://github.com/dfl-rlab/erc_sim_2026/blob/b1f9b05e20f4750b59f3321d88f172cc4dbb1386/README.md)
explicitly updates the book spine to 20 mm. The pinned hash includes that
notice and the merged model changes; it is not a newly named release.

The maintainer also [confirmed 20 mm books for the physical competition](https://github.com/dfl-rlab/erc_sim_2026/issues/2#issuecomment-5558021020)
and [retained position commands, suggesting closed-loop position control
using feedback rather than adding an effort command interface](https://github.com/dfl-rlab/erc_sim_2026/issues/2#issuecomment-5539799631).
Those published statements support using the official update without copying
custom controllers or simulator patches from another team.

## Exact update from historical v1.0.3

The [official comparison](https://github.com/dfl-rlab/erc_sim_2026/compare/0a09806ecbade5edc9f8a148b7c9f439ed761554...b1f9b05e20f4750b59f3321d88f172cc4dbb1386)
contains only README, book SDF and checked-in robot URDF changes.

| Property | Historical v1.0.3 | Current official b1f9b05 |
|---|---|---|
| Book local box dimensions | 0.25 × 0.03 × 0.16 m | 0.25 × 0.02 × 0.16 m |
| Book friction `mu` / `mu2` | 5 / 5 | 10 / 10 |
| Four fingertip friction pairs | 0.9 / 0.9 | 2.7 / 2.7 |
| Twelve inner/outer/fingertip hinge effort limits | 0.1 Nm | 40 Nm |
| Master screw effort limit | 10 N | Unchanged 10 N |
| Gripper command interface | Position with public clamp proxy | Unchanged |
| Controller plugin, torso and wheel settings | Supplied source | Unchanged |
| World maximum step | 0.002 s | Unchanged 0.002 s |
| Added book velocity decay | None | None |

The 40 Nm entries are revolute mimic-joint limits, not authorization to issue
a 40 N master command. The source generator is unchanged and does not
reproduce the updated checked-in URDF values. **Keep the official checked-in
URDF; do not regenerate or locally repair the robot model.**

The repository has adopted the exact official book/URDF files and adjusted
solution-level payload/grasp defaults for 20 mm books. Their identities are
recorded in [official_environment.json](../../official_environment.json).
The [manifest](official_b1f9b05_source_manifest.json) and [checker instructions](README.md)
cover all official source files, not just those two changed assets.

The manifest also pins the compiled payloads of the two tracked Python 3.10
cache files. An explicit checker option can accept logged header refreshes
from normal imports only when those payloads and magic bytes remain unchanged;
it does not exempt executable bytecode from checking.

## Evidence boundary

At this documentation update, a complete mission on the newly adopted
official model is pending. Earlier seed-101 trials, retry-19/25/26 poses,
saved route/grasp certificates and the 973-test historical result describe
v1.0.3 configurations. Preserve them as historical evidence and recompute
geometry and validate physical retention for this model. Recheck published
organizer updates before submission and record the exact source/image used;
neither source integrity nor test success is an official score.
