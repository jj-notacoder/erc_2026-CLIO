# Revision43 performance validation

Revision43 passed 2,801 full ROS tests with zero failures, errors or skips, selected/default four-node constructor checks, and unchanged official-asset checks. The canonical package was promoted only after validation and an exact 277-file comparison with the tested snapshot.

| Binding | SHA-256 |
|---|---|
| Package file-map digest | `3a7a082245544a63d258c8148f889bd5dcf7068da93c0905d7182156d71eb161` |
| final10 validation certificate | `e9f6ca713992df8e398d9ab9d4236a8c264ca894eb39ff3bab640a6aad3d1c64` |
| Selected runtime profile | `590bc458a136cd9d2ab9073104cd496077e5b2df65f0079ed4010a878dedd705` |

The package adds a sufficient separating-axis check before tool/body mesh intersection and caches exact local-mesh bounds by immutable content. Each separation check projects the complete current transformed vertices; uncertain cases retain the existing mesh predicate. Screen, table, bin, floor, retention and motion checks remain active. Local bounds are reused only for eligible array ownership and exact content; mutation or uncertain input uses the original computation. These changes do not change official models, controllers, physics, commanded grip, route sampling or motion duration.

In the fixed-context, profiled Run40 planning replay, wall time fell from **206.010 to 129.342 seconds (37.2%)**. The complete path, 2,337 sampled records, sampled verdict trace, 23 IK calls and one accepted candidate were identical. The baseline and candidate each had 106 exact scene-cache hits. This ordered offline comparison includes profiling overhead; it does not establish live mission speed or universal equivalence of the old conservative mesh test and the new sufficient separation proof.

The full evidence remains in the workspace under `output/official_reference/test_evidence/collision_quality_final_10`, `collision_quality_candidate/revision43`, and `run41_review/offline_profile/combined_comparison01`. The historical [Run40 physical result](revision42_validation.md#completed-run40-physical-result) belongs to revision42.

Run41 (`2dcdd10b8742`) started on these exact bytes at 21:29:32 UTC on 11 September, using the same seed101, column2/red profile and unchanged official assets. It secured the requested book, extracted it, completed the transport fold and returned to the bin. The PLACE command began at **955.742 observer wall seconds (15 min 55.742 s)**, already beyond the 15-minute full-mission target.

After capturing the diagnostic prefix, the agent sent one normal cancel at 21:48:27 UTC while the robot was stationary and holding the book. The node applied its existing hold/cancel path. The mission reported `ABORTED` at **1169.173200 observer wall seconds** with the generic reason `placement_failed`; the preserved command record identifies the initiating diagnostic cancellation. No PLACE motion, release, bin delivery or empty-hand return occurred. The observer completed its eight-ROS-second terminal tail at 1218.945685 wall seconds. The bin remained unchanged in saved poses. This run is a timing miss and a diagnostic partial run, not an autonomous grasp failure or a completed-mission benchmark.

The live profile covered the first **26 torso-validation scene calls** admitted within a 30-wall-second window. They consumed **10.798 profiled thread-CPU seconds**, including instrumentation overhead; 7.258 seconds were inside robot self-collision checks and 6.416 nested seconds inside mesh intersection. These inclusive times overlap, and this prefix is not the whole placement planner. Another observed 59.478-second PLACE worker interval used 16.280 CPU seconds; wall-minus-CPU does not identify a specific scheduling cause.

The bounded diagnostic itself adds overhead. In particular, after its time cutoff the wrapper repeatedly saved the same completed profile when further scene samples arrived; this is retained in the result and will be corrected in a later harness. Neither the live numbers nor the offline replay demonstrate the target yet.

The archived run contains 387 manifested files plus its manifest under `output/official_reference/runtime/run_41`. The CPU ledger stopped explicitly before archival. The observer, cameras, four team nodes, launch processes and Gazebo were stopped, and all eleven owned PIDs including the CPU sampler were verified absent. The shutdown record is SHA-256 `c2efcd1ed22ccd25a0b01358354630927845393393dc63b8f3edd9449e2b0537`. Official assets and the 277-file solution snapshot remained unchanged during the trial. The preceding Run40 remains the latest completed delivery evidence.
