# Column 4, blue — visible trial on 15 September 2026

**Result: ABORTED before grasping.** This is a failed trial, not a delivery.
The robot found printed marker 4, selected the blue book, and reached its grasp
standoff. After the normal row 3 head movement, the close-range metric tracker
withheld a fresh book point. The mission stopped with
`book_reacquisition_failed` after its 18.1-second simulation-time limit.

- Target model for seed 101: `book_col_5_row_4_blue`.
- Five navigation goals reached; zero grasp attempts and zero reported collisions.
- Solution launch to abort: 133.969 s; simulator startup to abort: 144.444 s.
- All 453 restored solution-source files remained unchanged.

The [camera image](camera_after_abort.png) shows the target still visible.
Replaying the unchanged estimator on the captured RGB-D arrays returned
`inconsistent_depth_geometry`. The dominant-face rule requires 60% support;
front-face hypotheses supplied 1,457 of 2,583 target points (56.4%). The viewing
angle itself was admissible. This is consistent with mixed front, side and top
surfaces, rather than an incorrect row or an invisible book.

This diagnosis uses one post-abort capture, not a recording of every frame
during the failed stage. No acceptance limit was relaxed.

[Mission summary](trial_d0fcc30b05eb_summary.json), [run review](run_review.json),
[offline result](camera_diagnostic.json), [stopped Gazebo view](stopped.png).

The original column 2/red 5m 18s mission is retained. Column 4/blue needs further
perception work before successful pickup or delivery can be claimed.
