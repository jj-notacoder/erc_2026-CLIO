# Test current source

The full suite requires **Python 3.10 and ROS 2 Humble**, installed message/description packages, NumPy, SciPy, OpenCV, PyYAML and pytest. Use the repository Docker/ROS workspace and its original sibling source layout. In particular, the tests need `erc_description`, `erc_bringup`, the TIAGo/omni-base/head/arm/gripper description sources, and `/opt/ros/humble/share/realsense2_description`. This supersedes the historical package README's claim that the complete suite runs without ROS.

After building and sourcing the workspace, run from the repository root:

```bash
./tools/run_solution_tests.sh
```

Optional pytest arguments are forwarded, for example:

```bash
./tools/run_solution_tests.sh -k installed_selected_profile
```

The launcher verifies two small [historical source fixtures](../tools/test_baselines/README.md), supplies `HEAD_RETURN_BASE_SOURCE` and `TORSO_RETRY_BASE_SOURCE`, and adds the package and test-support directories to `PYTHONPATH`. It uses one numerical-library thread and a separate default ROS domain (136; override with `ERC_TEST_ROS_DOMAIN_ID`). The fixtures are read as text/AST and are never installed as robot runtime code. Keep their bytes unchanged.

This checkout includes changes after candidate69. The historical counts below do not establish a full-suite result for current source. See [all-books validation](all_books_validation.md) for the scope of the recorded planning checks and physical trials; each result belongs to its recorded source version.

The launcher collects the current package tests. The two historical source files it supplies remain unchanged; no development archive is required. The separate historical combined63 two-case module is outside the package collection and is not claimed by this command. Do not run a heavy suite concurrently with a timed Gazebo experiment.

## Historical candidate69 baseline

The candidate69 / revision117 baseline contained a 453-file package snapshot. Its yaw change passed 534 affected tests and three fresh constructor modes. Its direct parent has a separate 659-test connector-fix result; an earlier 2,263-test affected result supplies navigation-module evidence. The 5,986-test Full63 result belongs to an ancestor. A complete suite was not rerun for that historical publication, and these counts must not be summed or described as a full run on candidate69 or current source. See the [compact baseline evidence](evidence/five_minute/README.md) for the separate records.
