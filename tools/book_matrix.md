# Reproducible book mission checks

Run inside the prepared ROS container, with the same workspace mounted at
`/opt/erc_ws`. The regular demo launcher must have completed the workspace and
Release controller builds first. Close any existing simulation before starting.

```bash
# All 20 requested printed-column/colour pairs, each in a fresh seed-101 world.
/entrypoint.sh /opt/erc_ws/tools/run_book_matrix.sh

# Show one target in Gazebo; the viewer closes after the delivery check.
/entrypoint.sh /opt/erc_ws/tools/run_book_matrix.sh \
  --targets 4:blue --seeds 101 --viewer

# A selected regression set; stop at the first failed check.
/entrypoint.sh /opt/erc_ws/tools/run_book_matrix.sh \
  --targets 4:blue 2:red --seeds 101 --stop-on-failure

# Repeat selected targets with other reproducible layouts.
/entrypoint.sh /opt/erc_ws/tools/run_book_matrix.sh \
  --targets 1:red 5:yellow --seeds 101 102 103
```

Columns refer to the **printed column marker requested by the mission**, not the
physical column index embedded in Gazebo's model name. The delivery evaluator
uses the exact `target_book_model` confirmed in each mission summary.

Each trial uses the existing simulation launch with `headless:=true` and
`depth_cloud:=false`, the selected collision-quality solution profile, and the
existing Release controller overlay. The optional viewer uses the regular local
viewer configuration. Sensor rates, physics, mission guards, and robot commands
are unchanged by this runner. Headless timing does not establish timing with a
visible viewer or a recording.

`--timeout` bounds each world from immediately before simulator startup; the
default is 900 wall seconds, with an allowed range of 60–1800. Readiness is
separately limited to 60 seconds. After a terminal mission summary, the
independent delivery sample and process cleanup can add up to approximately
44 seconds. Ctrl+C stops the current trial and all process groups owned by the
runner. Cleanup escalates from SIGINT to SIGTERM to SIGKILL when needed, checks
for surviving descendants, and refuses to start another world if cleanup fails.
It never broadly kills other Gazebo or ROS processes.

Results are written under `results/matrix_<UTC timestamp>/`:

- `matrix_summary.json` is updated after every trial, including failures.
- Each trial keeps simulation, solution, readiness, and diagnostic logs, the
  ordinary mission JSONL and summary, process arguments/PIDs, and its source
  manifest. Source files and the selected Release controller libraries must
  remain unchanged across the matrix. Python caches and tooling are excluded
  from this freeze check. Stop the runner before changing mission code.
- `perception_status.jsonl` captures both perception status topics, including
  edge-triggered tracking failure reasons, producer timestamps when available,
  receipt monotonic time, UTC, and the recorder's ROS clock. This recorder only
  subscribes; it starts before the solution.
- `run_review.json` distinguishes simulator-start timing, solution-process-start
  timing (including ROS CLI startup), and the solution's own reported launch
  origin. All end at the mission's terminal summary; evaluation is excluded.
- `containment.json` is sampled only after a successful physical mission. A
  validated result requires exact requested-target confirmation, zero recorded
  collision episodes, and the whole book inside the evaluator's conservative
  bin core. `DONE` alone is insufficient. A negative clearance, even a tiny
  numerical one, remains a failed strict check.

The matrix exits successfully only when every requested trial has a validated
delivery with unchanged source and complete process cleanup. A seed-101 sweep
checks those 20 layouts; it does not prove success for every random seed or
long-term stability after the single independent containment sample.

Runner tests require only Python's standard library:

```bash
python3 -m unittest discover -s tools -p test_book_matrix.py -v
```
