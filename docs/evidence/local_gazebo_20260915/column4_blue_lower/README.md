# Column 4 blue: first lower-row physical attempt

Source `9690fa8`, trial `180fb89230d6`, seed 101, visible Gazebo.

**ABORTED; no acquired book or delivery.** The close-view fit succeeded in
0.3 simulated seconds. Complete nominal lower-row planning passed, but the
empty setup lacked full shelf clearance checks. Arm link 6 touched the shelf
during entry. Closure stopped for unilateral contact. The reverse recovery
also touched the shelf with arm link 5. Two collision episodes were recorded.

The run took 317.413 wall seconds from solution launch, or 327.678 seconds
including simulator startup. All owned processes stopped normally and source
remained unchanged. These times end at ABORTED and are not mission-completion
times. This route is being corrected before another physical trial.

- [Mission summary](trial_180fb89230d6_summary.json)
- [Independent run review](run_review.json)
- [Recorded mission events](trial_180fb89230d6.jsonl)
- [Pickup view](pickup_view.png)

The nominal fit and IK diagnostics are estimates and planned poses. They do
not establish measured alignment after the shelf contact. The journal omitted
some empty-arm diagnostic events; subsequent runs retain those events and
their parked-joint context.
