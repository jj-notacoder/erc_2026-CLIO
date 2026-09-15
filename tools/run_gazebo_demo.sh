#!/usr/bin/env bash
# Run inside the prepared ROS container, through /entrypoint.sh.
set -e
demo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$demo_root"
if pgrep -f '[r]os2 launch erc_bringup simulation.launch.py' >/dev/null; then
  echo 'A simulator is already running in this container. Close that run first.' >&2
  exit 1
fi
if [ -z "${DISPLAY:-}" ]; then
  echo 'DISPLAY must point to your desktop to show Gazebo.' >&2
  exit 1
fi
source /opt/ros/humble/setup.bash
# Setup files can be generated before colcon fails. Reuse an install only
# after a completed build, and check the artifacts needed by this launcher.
demo_main_marker=install/.erc_demo_build_complete_v1
demo_release_marker=install_release/.erc_demo_build_complete_v1
demo_main_artifacts_ready() {
  local demo_node
  [ -f install/setup.bash ] &&
    [ -f install/erc_bringup/share/erc_bringup/launch/simulation.launch.py ] &&
    [ -f install/erc_phase1_solution/share/erc_phase1_solution/launch/solution.launch.py ] || return 1
  for demo_node in perception_node navigation_node manipulation_node mission_manager; do
    [ -x "install/erc_phase1_solution/lib/erc_phase1_solution/$demo_node" ] || return 1
  done
}
demo_release_artifacts_ready() {
  [ -f install_release/local_setup.bash ] &&
    [ -f install_release/gz_ros2_control/lib/libgz_ros2_control-system.so ] &&
    [ -f install_release/gz_ros2_control/lib/libgz_hardware_plugins.so ]
}
if [ ! -f "$demo_main_marker" ] || ! demo_main_artifacts_ready; then
  # A changed or incomplete underlay also requires a fresh Release overlay.
  rm -f -- "$demo_main_marker" "$demo_release_marker"
  colcon build --symlink-install --parallel-workers 3 \
    --cmake-args -DCMAKE_BUILD_TYPE=Release
  if ! demo_main_artifacts_ready; then
    echo 'Workspace build returned without the required installed launch files or executables.' >&2
    exit 1
  fi
  source install/setup.bash
  touch "$demo_main_marker"
else
  source install/setup.bash
fi
if [ ! -f "$demo_release_marker" ] || ! demo_release_artifacts_ready; then
  rm -f -- "$demo_release_marker"
  MAKEFLAGS=-j1 colcon --log-base log_release build \
    --build-base build_release --install-base install_release \
    --symlink-install --packages-select gz_ros2_control --parallel-workers 1 \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
  if ! demo_release_artifacts_ready; then
    echo 'Release build returned without the required controller libraries.' >&2
    exit 1
  fi
  source install_release/local_setup.bash
  touch "$demo_release_marker"
else
  source install_release/local_setup.bash
fi
cmake -S tools/gazebo_viewer_plugin -B tools/gazebo_viewer_plugin/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build tools/gazebo_viewer_plugin/build --parallel 2
export GZ_GUI_PLUGIN_PATH="$demo_root/tools/gazebo_viewer_plugin/build${GZ_GUI_PLUGIN_PATH:+:$GZ_GUI_PLUGIN_PATH}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-31}"
export GZ_PARTITION="${GZ_PARTITION:-erc_local_demo}"
export ERC_SEED="${ERC_SEED:-101}"
export ERC_RESULTS_DIR="$demo_root/results/demo_$(date -u +%Y%m%dT%H%M%SZ)"
export ERC_ERC_IMAGES_DIR="$demo_root/erc_images"
mkdir -p "$ERC_RESULTS_DIR"
demo_pids=()
cleanup() {
  trap - EXIT INT TERM
  for demo_pid in "${demo_pids[@]}"; do
    kill -INT -- "-$demo_pid" 2>/dev/null || true
  done
  wait || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python3 - <<'PY_START'
import json, os, time
from pathlib import Path
from datetime import datetime, timezone
Path(os.environ['ERC_RESULTS_DIR'], 'simulator_start.json').write_text(json.dumps({
    'utc': datetime.now(timezone.utc).isoformat(timespec='microseconds'),
    'monotonic_seconds': time.monotonic(),
    'basis': 'before_simulator_and_viewer_launch',
}, indent=2) + '\n')
PY_START
setsid ros2 launch erc_bringup simulation.launch.py \
  headless:=true depth_cloud:=false >"$ERC_RESULTS_DIR/simulation.log" 2>&1 &
demo_pids+=("$!")
setsid gz sim -g --render-engine-gui ogre \
  --gui-config "$demo_root/tools/gazebo_viewer.config" \
  >"$ERC_RESULTS_DIR/viewer.log" 2>&1 &
demo_pids+=("$!")
python3 tools/check_sensor_readiness.py
echo "Starting shelf ${ERC_SHELF_COLUMN:-2}, ${ERC_BOOK_COLOUR:-red}; logs: $ERC_RESULTS_DIR"
setsid ros2 launch erc_phase1_solution solution.launch.py \
  shelf_column_number:="${ERC_SHELF_COLUMN:-2}" \
  book_colour:="${ERC_BOOK_COLOUR:-red}" \
  >"$ERC_RESULTS_DIR/solution.log" 2>&1 &
demo_pids+=("$!")
# Mission DONE keeps its ROS launch alive. Surface crashed launch processes,
# while a normal viewer close (or Ctrl+C) closes this run's process groups.
if wait -n "${demo_pids[@]}"; then demo_status=0; else demo_status=$?; fi
if ! kill -0 "${demo_pids[1]}" 2>/dev/null; then
  exit "$demo_status"
elif ! kill -0 "${demo_pids[2]}" 2>/dev/null; then
  echo 'The solution launch exited. Its last messages are:' >&2
  tail -n 30 "$ERC_RESULTS_DIR/solution.log" >&2
else
  echo 'The simulator exited. Its last messages are:' >&2
  tail -n 30 "$ERC_RESULTS_DIR/simulation.log" >&2
fi
exit "$((demo_status == 0 ? 1 : demo_status))"
