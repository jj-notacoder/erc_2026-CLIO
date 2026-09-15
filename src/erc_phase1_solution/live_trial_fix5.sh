#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /opt/erc_ws/install/setup.bash
export ROS_DOMAIN_ID=23

case "${1:-}" in
  sim)
    export ERC_SEED=101
    export GZ_SIM_RESOURCE_PATH=/opt/erc_ws/install/erc_description/share:/opt/erc_ws/install/omni_base_description/share:/opt/erc_ws/install/tiago_pro_description/share:/opt/erc_ws/install/pal_sea_arm_description/share:/opt/erc_ws/install/tiago_pro_head_description/share:/opt/erc_ws/install/pal_pro_gripper_description/share:/opt/erc_ws/install/pal_gripper_description/share:/opt/erc_ws/install/pal_urdf_utils/share:/opt/ros/humble/share
    export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/erc_ws/install/gz_ros2_control/lib
    echo $$ >/tmp/erc_sim_fix16.pgid
    cd /opt/erc_ws
    exec ros2 launch erc_bringup simulation.launch.py headless:=true >results/sim_seed101_fix16.log 2>&1
    ;;
  solution)
    echo $$ >/tmp/erc_solution_fix16.pgid
    cd /opt/erc_ws
    exec ros2 launch erc_phase1_solution solution.launch.py shelf_column_number:=2 book_colour:=red trial_timeout_seconds:=1200 rviz:=false >results/solution_seed101_fix16.log 2>&1
    ;;
  *)
    echo "usage: $0 {sim|solution}" >&2
    exit 2
    ;;
esac
