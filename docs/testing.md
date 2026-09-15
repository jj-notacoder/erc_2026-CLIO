# Test the candidate69 release

The full suite requires **Python 3.10 and ROS 2 Humble**, installed message/description packages, NumPy, SciPy, OpenCV, PyYAML and pytest. Use the repository Docker/ROS workspace and its original sibling source layout. In particular, the tests need `erc_description`, `erc_bringup`, the TIAGo/omni-base/head/arm/gripper description sources, and `/opt/ros/humble/share/realsense2_description`. This supersedes the historical package README's claim that the complete suite runs without ROS.

After building and sourcing the workspace, run from the repository root:

```bash
./tools/run_solution_tests.sh
```

Optional pytest arguments are forwarded, for example:

```bash
./tools/run_solution_tests.sh -k installed_selected_profile
```

The launcher verifies two small [historical source fixtures](../tools/test_baselines/README.md), supplies `HEAD_RETURN_BASE_SOURCE` and `TORSO_RETRY_BASE_SOURCE`, and adds the package and test-support directories to `PYTHONPATH`. It uses one numerical-library thread and a separate default ROS domain (234; override with `ERC_TEST_ROS_DOMAIN_ID`). The fixtures are read as text/AST and are never installed as robot runtime code. Keep their bytes unchanged.

This release contains the exact 453-file candidate69 / revision117 package. The current yaw change passed 534 affected tests and three fresh constructor modes. Its direct parent has a separate 659-test connector-fix result; an earlier 2,263-test affected result supplies navigation-module evidence. The 5,986-test Full63 result belongs to an ancestor. A complete suite was not rerun for this publication, and these counts must not be summed or described as a full run on candidate69. See the [compact evidence](evidence/five_minute/README.md) for the separate records.

All current package test fixtures are included in the 453-file source snapshot. The two historical source files supplied by the launcher remain unchanged; no development archive is required. The separate historical combined63 two-case module is outside the package collection and is not claimed by this command. Do not run a heavy suite concurrently with a timed Gazebo experiment.
