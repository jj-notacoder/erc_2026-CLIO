#!/usr/bin/env bash
# Run inside the prepared ROS container, through /entrypoint.sh.
set -e
matrix_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$matrix_root"
if [[ " ${*} " == *" --help "* || " ${*} " == *" -h "* ]]; then
  exec python3 tools/run_book_matrix.py "$@"
fi
for matrix_artifact in \
  install/.erc_demo_build_complete_v1 \
  install/erc_bringup/share/erc_bringup/launch/simulation.launch.py \
  install/erc_phase1_solution/share/erc_phase1_solution/launch/solution.launch.py \
  install_release/.erc_demo_build_complete_v1 \
  install_release/local_setup.bash \
  install_release/gz_ros2_control/lib/libgz_ros2_control-system.so \
  install_release/gz_ros2_control/lib/libgz_hardware_plugins.so; do
  if [ ! -f "$matrix_artifact" ]; then
    echo "Prepared build missing: $matrix_artifact. Build with tools/run_gazebo_demo.sh first." >&2
    exit 1
  fi
done
source /opt/ros/humble/setup.bash
source install/setup.bash
source install_release/local_setup.bash
if [[ " ${*} " == *" --viewer "* ]]; then
  if [ -z "${DISPLAY:-}" ]; then
    echo 'DISPLAY must point to your desktop when --viewer is selected.' >&2
    exit 1
  fi
  cmake -S tools/gazebo_viewer_plugin -B tools/gazebo_viewer_plugin/build \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build tools/gazebo_viewer_plugin/build --parallel 2
  export GZ_GUI_PLUGIN_PATH="$matrix_root/tools/gazebo_viewer_plugin/build${GZ_GUI_PLUGIN_PATH:+:$GZ_GUI_PLUGIN_PATH}"
fi
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-31}"
export GZ_PARTITION="${GZ_PARTITION:-erc_book_matrix}"
exec python3 tools/run_book_matrix.py "$@"
