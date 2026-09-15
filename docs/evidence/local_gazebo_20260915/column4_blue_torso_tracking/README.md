# Blue PLACE torso tracking failure

Visible seed-101 trial `8d722ba0fcbf`, printed column 4/blue, used frozen source
`014bbdc`. It reached complete PLACE planning, nominal torso action success and
the restored 0.8 s first arm setup movement. The measured stationary handoff
failed before the second arm leg. No opening, release or delivery occurred.

- Exact target: `book_col_5_row_4_blue`; zero collision episodes.
- Solution process launch to terminal: **505.302 s**; reported launch origin:
  **505.112 s**; simulator startup to terminal: **515.275 s**.
- All source hashes stayed unchanged and all owned process groups stopped.
- Handoff used its unchanged **0.5 simulated-second** settling window;
  observed wall time was **2.9268 s**.

## Measured cause

All 125 sampled predicates failed. The torso reference was **0.35 m**. Its largest
position error was **0.137886 m**; the last sample was still **0.120526 m below**
reference and rising at approximately **0.035 m/s**. In the last sample, every arm
joint position error was below 0.000001 rad and arm speed below 0.000012 rad/s.
Base motion and gripper motion were also small. Thus extending the first arm
movement did not address the persistent torso error.

The normal PLACE caller had requested the rise from approximately 0.10 to
0.35 m in 2.2 s. Controller action success did not establish arrival at the
commanded height. The next correction must use a feasible torso duration and
measured completion before dispatching the loaded arm route. The stationary
handoff correctly rejected the unachieved torso endpoint.

## Evidence and limits

`selected_journal_events.jsonl` preserves the complete `placement_transition_stop`
event, its aggregate values and last 12 raw samples. `recorded_place_inputs.json`
and `recorded_place_plan.json` preserve the exact selected planning values.
The matching perception observation was uniquely identified by full bin/table
scene equality, at stamp 112334000000 ns. Accepted joint waypoint arrays are not
logged. This record establishes the measured torso failure, not physical release
or valid execution of the later plan.

Screenshots are unmodified visible Gazebo captures, visually inspected. The
terminal image is the watcher's last visible frame before GUI cleanup; its actual
capture time and fallback label are retained in `screenshot_metadata.json`.
