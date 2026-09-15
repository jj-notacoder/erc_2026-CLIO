# Run the simulation from this repository

Keep this whole repository together. Its `src/` contains the solution, robot
descriptions, Gazebo world and controller sources; `tools/` and `docker/` provide
the visible demonstration. Generated `build*`, `install*`, `log*`, `results/`
and `erc_images/` directories are local outputs and do not need to be pushed.

## Linux desktop

Use an x86-64 Linux desktop with Docker Engine, Python 3, and X11. Docker must be
usable by your account. Run from a desktop terminal so `DISPLAY` and
`XAUTHORITY` identify the active session. Intel/AMD graphics at `/dev/dri` are
passed into the container. Other graphics setups can use the existing Docker
Compose GPU/WSLg instructions in the main README.

From the repository root:

```bash
./tools/show_gazebo.sh --check 4 blue
./tools/show_gazebo.sh 4 blue
```

The request is the **numbered column marker**, whose physical position is
randomized. The defaults are `2 red`; accepted columns are 1–5 and colours are
`red`, `blue`, `green`, and `yellow`.

On first use, the script builds the Docker image if its tag is absent. The
container launcher then builds the workspace, the Release controller overlay,
and the viewer plugin as needed. These first-time builds happen before the
mission starts and can take several minutes. A compatible `erc_friend_demo`
container is reused. A container for another checkout is left intact and a
separate container name is chosen.
Successful builds receive completion markers in `install/` and
`install_release/`. An interrupted build is retried on the next launch. The
first launch after upgrading an older checkout may rebuild its existing cache
once to establish these markers.
The launcher refuses to start a second simulation in a known ERC container,
so another live run cannot compete for CPU and GPU time. Alternate container
names derive a deterministic ROS domain in the range 40–89.

The Gazebo window shows the live mission. Close it or press Ctrl+C in the
launching terminal to stop the run. Simulation, viewer and solution logs are
written under `results/demo_<timestamp>/`; the mission writes its summary there.
The optional host performance profile lasts only while the launcher runs.

```bash
# Same setup and target, another deterministic world seed:
ERC_SEED=102 ./tools/show_gazebo.sh 4 blue

# Use the current host power profile:
ERC_USE_PERFORMANCE=0 ./tools/show_gazebo.sh 2 red
```

The Docker source includes the local CycloneDDS configuration used for the
working demonstrations, plus explicit scientific Python and viewer build
dependencies. To rebuild an existing image after changing the Docker files:

```bash
docker build -t erc-2026:humble-harmonic -f docker/Dockerfile .
ERC_CONTAINER_NAME=erc_demo_rebuilt ./tools/show_gazebo.sh 4 blue
```

Use the rebuilt container name on subsequent runs. Builds reuse the source
mounted at `/opt/erc_ws`; do not edit mission source while a trial is active.

## GitHub

Push the repository sources, launch tools, documentation and required robot
assets. The root `.gitignore` excludes generated build outputs and trial logs.
The project should run after cloning without copying another checkout's build
directories. Keep license and third-party notice files with the sources.

The successful local red-book baseline took approximately 5m 18s from solution
launch to completion, or 5m 29s including simulator startup. Timing depends on
the host, scene and requested book; this is not an under-five-minute guarantee.

The visible column 4/blue trial found the target but aborted before grasping
when the unchanged depth tracker rejected its surface geometry. It is not a
verified delivery target yet. [Trial evidence](evidence/local_gazebo_20260915/column4_blue/README.md).

## Packaging validation

A clean export of the staged Git files matched all **887 official simulator
source files** byte-for-byte and all **453 restored solution-source files**.
It contains no generated build directories or raw run-log directories.
[Source check](evidence/local_gazebo_20260915/github_source_check.json).

The host launcher passed shell syntax, argument, container compatibility and
second-simulation rejection checks. Mocked build checks covered interrupted
build retries and incomplete cache detection. The existing prepared container
ran the visible trial above; a fresh Docker image rebuild was not rerun during
this cleanup.
