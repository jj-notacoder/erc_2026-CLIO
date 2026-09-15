# Optimized controller plugin build

The original `build/gz_ros2_control/CMakeCache.txt` had an empty build type;
generated compiler flags contained no optimization flag. This separate Release
build uses `-O3 -DNDEBUG` on the existing source without replacing libraries
mapped by an active simulation.

From the host:

```bash
docker exec -w /opt/erc_ws -e MAKEFLAGS=-j1 erc_friend_demo /entrypoint.sh bash -c '
  source /opt/erc_ws/install/setup.bash &&
  exec nice -n 15 colcon --log-base log_release build \
    --build-base build_release --install-base install_release --symlink-install \
    --packages-select gz_ros2_control --parallel-workers 1 \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
'
```

For the next simulation process, source the release overlay after the original
workspace. The following environment setup belongs inside the entrypoint shell:

```bash
source /opt/erc_ws/install/setup.bash
source /opt/erc_ws/install_release/local_setup.bash
export GZ_SIM_SYSTEM_PLUGIN_PATH="/opt/erc_ws/install_release/gz_ros2_control/lib${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"
export LD_LIBRARY_PATH="/opt/erc_ws/install_release/gz_ros2_control/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The overlay's ament package index also selects its `gz_hardware_plugins` library.
Keep `GZ_SIM_RESOURCE_PATH` from the entrypoint: all world, robot, and sensor
assets still come from the original workspace. No asset overlay is needed.
The existing simulation keeps using its already loaded original libraries.

Checks before launching:

```bash
ros2 pkg prefix gz_ros2_control
grep '^CXX_FLAGS' /opt/erc_ws/build_release/gz_ros2_control/CMakeFiles/gz_ros2_control-system.dir/flags.make
ldd -r /opt/erc_ws/install_release/gz_ros2_control/lib/libgz_ros2_control-system.so
ldd -r /opt/erc_ws/install_release/gz_ros2_control/lib/libgz_hardware_plugins.so
```

Expected package prefix: `/opt/erc_ws/install_release/gz_ros2_control`; expected
compiler flags include `-O3 -DNDEBUG`. After launching, the server's `/proc/PID/maps`
should show both libraries from `build_release/gz_ros2_control` (the install
directory contains symlinks). Performance improvement requires a new simulation
and a measured comparison; compiling alone does not establish a speedup.
