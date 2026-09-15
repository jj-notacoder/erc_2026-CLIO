"""Competition entry point for the ERC 2026 Phase 1 solution."""

import time
from datetime import datetime, timezone

# Capture before importing ROS launch or building nodes. This excludes the
# ros2 CLI/Python startup before this file is evaluated, which is not observable
# here; callers with an earlier origin may explicitly override all three fields.
_LAUNCH_ORIGIN_MONOTONIC_SECONDS = time.monotonic()
_LAUNCH_ORIGIN_UTC = datetime.now(timezone.utc).isoformat(timespec='microseconds')
_LAUNCH_ORIGIN_BASIS = 'solution.launch.py_module_entry_before_ros_imports'

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from erc_phase1_solution.runtime_utils import validate_task_parameters


def _writable_output(name: str) -> str:
    """Return a stable trial-output path rooted in the current team checkout."""
    override = os.environ.get(f'ERC_{name.upper()}_DIR')
    if override:
        return os.path.abspath(override)

    cwd = Path.cwd().resolve()
    roots = [
        candidate
        for candidate in (cwd, *cwd.parents)
        if (candidate / '.git').exists()
        and (candidate / 'src' / 'erc_phase1_solution' / 'package.xml').exists()
    ]
    official_workspace = Path('/opt/erc_ws')
    if (
        official_workspace not in roots
        and (
            official_workspace
            / 'src'
            / 'erc_phase1_solution'
            / 'package.xml'
        ).exists()
    ):
        roots.append(official_workspace)
    if not roots:
        roots.append(cwd)

    failures = []
    for root in roots:
        candidate = root / name
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / '.write_probe'
            probe.touch(exist_ok=True)
            probe.unlink(missing_ok=True)
            return str(candidate)
        except OSError as exc:
            failures.append(f'{candidate}: {exc}')
    raise RuntimeError(
        f'No writable {name!r} output directory: ' + '; '.join(failures)
    )


def _launch_nodes(context):
    column_raw = LaunchConfiguration('shelf_column_number').perform(context).strip()
    try:
        column, colour = validate_task_parameters(
            column_raw,
            LaunchConfiguration('book_colour').perform(context),
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    share = Path(get_package_share_directory('erc_phase1_solution'))
    config = str(share / 'config' / 'solution.yaml')
    profile = LaunchConfiguration('profile', default='collision_quality').perform(context).strip()
    if profile not in ('default', 'collision_quality'):
        raise RuntimeError('Unknown profile: ' + profile)
    parameter_files = [config]
    numeric_environment = {}
    if profile == 'collision_quality':
        profile_file = share / 'config' / 'collision_quality.yaml'
        if not profile_file.is_file():
            raise RuntimeError('Collision-quality profile is not installed: ' + str(profile_file))
        parameter_files.append(str(profile_file))
        numeric_environment = {
            'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    templates = str(share / 'assets' / 'digits')
    common = [
        *parameter_files,
        {
            'shelf_column_number': column,
            'book_colour': colour,
            'digit_template_dir': templates,
            'image_output_dir': _writable_output('erc_images'),
            'results_output_dir': _writable_output('results'),
            'team_name': LaunchConfiguration('team_name'),
            'trial_timeout_seconds': ParameterValue(
                LaunchConfiguration('trial_timeout_seconds'), value_type=float
            ),
            'dry_run': ParameterValue(
                LaunchConfiguration('dry_run'), value_type=bool
            ),
        },
    ]

    return [
        Node(
            package='erc_phase1_solution',
            executable='perception_node',
            name='erc_perception',
            output='screen',
            parameters=common,
            additional_env=numeric_environment,
        ),
        Node(
            package='erc_phase1_solution',
            executable='navigation_node',
            name='erc_navigation',
            output='screen',
            parameters=common,
            additional_env=numeric_environment,
        ),
        Node(
            package='erc_phase1_solution',
            executable='manipulation_node',
            name='erc_manipulation',
            output='screen',
            parameters=common,
            additional_env=numeric_environment,
        ),
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='erc_phase1_solution',
                    executable='mission_manager',
                    name='erc_mission_manager',
                    output='screen',
                    additional_env=numeric_environment,
                    parameters=[
                        *parameter_files,
                        {
                            **common[-1],
                            'launch_origin_monotonic_seconds': ParameterValue(
                                LaunchConfiguration(
                                    'launch_origin_monotonic_seconds', default='-1.0'
                                ), value_type=float,
                            ),
                            'launch_origin_utc': ParameterValue(
                                LaunchConfiguration('launch_origin_utc', default=''),
                                value_type=str,
                            ),
                            'launch_origin_basis': ParameterValue(
                                LaunchConfiguration(
                                    'launch_origin_basis', default='not_provided'
                                ), value_type=str,
                            ),
                        },
                    ],
                )
            ],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='erc_solution_rviz',
            arguments=['-d', str(share / 'config' / 'solution.rviz')],
            condition=IfCondition(LaunchConfiguration('rviz')),
            output='screen',
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'shelf_column_number',
                description='Requested randomized shelf-column marker (1-5).',
            ),
            DeclareLaunchArgument(
                'book_colour',
                description='Requested book colour: red, blue, green, or yellow.',
            ),
            DeclareLaunchArgument(
                'team_name',
                default_value='TEAM_NAME',
                description='Team name recorded in mission status and trial logs.',
            ),
            DeclareLaunchArgument(
                'profile', default_value='collision_quality',
                description=(
                    'collision_quality selects the installed competition configuration. '
                    'Use profile:=default explicitly for the base solution.yaml settings.'
                ),
            ),
            DeclareLaunchArgument('rviz', default_value='false'),
            DeclareLaunchArgument(
                'launch_origin_monotonic_seconds',
                default_value=repr(_LAUNCH_ORIGIN_MONOTONIC_SECONDS),
                description=(
                    'Host monotonic launch origin, including delayed node startup. '
                    'An override must use the same host monotonic clock as the node '
                    'and supply matching launch_origin_utc and launch_origin_basis.'
                ),
            ),
            DeclareLaunchArgument(
                'launch_origin_utc',
                default_value=_LAUNCH_ORIGIN_UTC,
                description='UTC ISO timestamp captured alongside the launch origin.',
            ),
            DeclareLaunchArgument(
                'launch_origin_basis',
                default_value=_LAUNCH_ORIGIN_BASIS,
                description=(
                    'Exact origin capture point. The default excludes CLI startup '
                    'before this launch file is evaluated.'
                ),
            ),
            DeclareLaunchArgument(
                'trial_timeout_seconds',
                default_value='1800.0',
                description=(
                    'Development wall-clock watchdog matching the selected launch. '
                    'This allowance does not satisfy the five-minute performance target; '
                    'elapsed time must be evaluated separately.'
                ),
            ),
            DeclareLaunchArgument(
                'dry_run',
                default_value='false',
                description='Exercise perception and planning without commanding motion.',
            ),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
