"""Regression coverage for evaluator-facing launch parameter precedence."""

from importlib.util import module_from_spec, spec_from_file_location
import math
from pathlib import Path

import pytest
import yaml


launch = pytest.importorskip('launch')
if not hasattr(launch, 'LaunchContext'):
    pytest.skip(
        'ROS 2 launch is unavailable in this host Python environment',
        allow_module_level=True,
    )
LaunchContext = launch.LaunchContext


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_FILE = PACKAGE_ROOT / 'launch' / 'solution.launch.py'


def _load_launch_module():
    spec = spec_from_file_location('erc_solution_launch_test', LAUNCH_FILE)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_trial_timeout_has_a_single_launch_controlled_source(
    monkeypatch, tmp_path
):
    """An exact-node YAML value must not mask the launch argument on Humble."""
    static_config = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'solution.yaml').read_text(encoding='utf-8')
    )
    mission_config = static_config['erc_mission_manager']['ros__parameters']
    assert 'trial_timeout_seconds' not in mission_config

    launch_module = _load_launch_module()
    monkeypatch.setattr(
        launch_module,
        'get_package_share_directory',
        lambda _package: str(PACKAGE_ROOT),
    )
    monkeypatch.setattr(
        launch_module,
        '_writable_output',
        lambda name: str(tmp_path / name),
    )

    context = LaunchContext()
    context.launch_configurations.update(
        {
            'shelf_column_number': '2',
            'book_colour': 'red',
            'team_name': 'TEST_TEAM',
            'trial_timeout_seconds': '1200',
            'dry_run': 'true',
            'rviz': 'false',
        }
    )

    actions = launch_module._launch_nodes(context)
    mission_node = actions[3].actions[0]
    mission_node._perform_substitutions(context)
    parameter_arguments = mission_node._Node__expanded_parameter_arguments

    common_override_path, is_file = parameter_arguments[-1]
    assert is_file is True
    common_overrides = yaml.safe_load(
        Path(common_override_path).read_text(encoding='utf-8')
    )
    assert common_overrides['/**']['ros__parameters'][
        'trial_timeout_seconds'
    ] == 1200.0


def test_retreat_goal_guarantees_the_manipulation_shelf_clearance():
    """Cross-node defaults must not overstate achieved shelf-normal retreat."""
    static_config = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'solution.yaml').read_text(encoding='utf-8')
    )
    navigation = static_config['erc_navigation']['ros__parameters']
    manipulation = static_config['erc_manipulation']['ros__parameters']
    mission = static_config['erc_mission_manager']['ros__parameters']

    guaranteed_normal_retreat = (
        mission['carried_shelf_retreat_distance']
        - navigation['position_tolerance']
    ) * math.cos(navigation['yaw_tolerance']) - manipulation[
        'carried_navigation_radius_limit'
    ] * math.sin(navigation['yaw_tolerance'])
    assert guaranteed_normal_retreat > 0.0
    assert manipulation['carried_shelf_retreat_clearance_distance'] <= (
        guaranteed_normal_retreat
    )


def test_carried_retreat_profile_is_bounded_and_brakes_before_tolerance():
    static_config = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'solution.yaml').read_text(encoding='utf-8')
    )
    navigation = static_config['erc_navigation']['ros__parameters']

    assert 0.0 < navigation['carried_retreat_max_speed'] <= min(
        navigation['max_linear_speed'],
        navigation['max_lateral_speed'],
    )
    assert (
        0.0
        < navigation['carried_retreat_braking_acceleration']
        <= navigation['carried_retreat_acceleration_limit']
        <= navigation['linear_acceleration_limit']
    )
