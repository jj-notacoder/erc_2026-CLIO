"""Expand actual Humble launch actions and parameter files; execute no process."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import yaml

launch=pytest.importorskip('launch')
if not hasattr(launch,'LaunchContext'):
    pytest.skip('ROS 2 launch is unavailable',allow_module_level=True)
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.utilities import perform_substitutions

PACKAGE=Path(__file__).resolve().parents[1]
THREADS={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}


def _module(monkeypatch,tmp_path):
    spec=spec_from_file_location('profile_launch_under_test',PACKAGE/'launch/solution.launch.py')
    module=module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module,'get_package_share_directory',lambda _:str(PACKAGE))
    monkeypatch.setattr(module,'_writable_output',lambda name:str(tmp_path/name))
    def forbidden(*args,**kwargs):
        raise AssertionError('A launch expansion test must never execute a process')
    monkeypatch.setattr(ExecuteProcess,'execute',forbidden)
    return module


def _scoped_parameters(files,node_name):
    """Read the real files emitted by launch in their actual CLI order."""
    merged={}
    documents=[]
    for path,is_file in files:
        assert is_file
        document=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        documents.append(document)
        for scope in ('/**',node_name,'/'+node_name):
            merged.update(document.get(scope,{}).get('ros__parameters',{}))
    return merged,documents


@pytest.mark.parametrize('selection',[None,'default','collision_quality'])
def test_real_launch_profile_expansion_and_cli_precedence(monkeypatch,tmp_path,selection):
    module=_module(monkeypatch,tmp_path)
    context=launch.LaunchContext()
    context.launch_configurations.update(dict(shelf_column_number='2',book_colour='red',
        team_name='PROFILE_TEST',trial_timeout_seconds='1234.5',dry_run='true',rviz='false',
        launch_origin_monotonic_seconds='90.0',launch_origin_utc='2026-09-11T00:00:00+00:00',
        launch_origin_basis='test_external_origin'))
    if selection is not None:context.launch_configurations['profile']=selection
    declarations=module.generate_launch_description().entities
    for action in declarations:
        if isinstance(action,DeclareLaunchArgument):action.execute(context)
    assert context.launch_configurations['profile']==(selection or 'collision_quality')
    actions=module._launch_nodes(context)
    nodes=[*actions[:3],actions[3].actions[0]]
    names=('erc_perception','erc_navigation','erc_manipulation','erc_mission_manager')
    selected=selection in (None, 'collision_quality')
    resolved={}
    for node,name in zip(nodes,names):
        node._perform_substitutions(context)
        files=node._Node__expanded_parameter_arguments
        assert Path(files[0][0])==PACKAGE/'config/solution.yaml'
        assert len(files)==(3 if selected else 2)
        if selected:assert Path(files[1][0])==PACKAGE/'config/collision_quality.yaml'
        actual,documents=_scoped_parameters(files,name)
        assert actual['shelf_column_number']==2 and actual['book_colour']=='red'
        assert actual['team_name']=='PROFILE_TEST' and actual['dry_run'] is True
        assert actual['trial_timeout_seconds']==1234.5
        # The launch override file is final, and the profile cannot add a competing exact-node timeout.
        assert documents[-1]['/**']['ros__parameters']['trial_timeout_seconds']==1234.5
        if selected:
            assert all('trial_timeout_seconds' not in entry['ros__parameters']
                       for entry in documents[1].values())
        environment={perform_substitutions(context,key):perform_substitutions(context,value)
                     for key,value in (node.additional_env or [])}
        assert environment==(THREADS if selected else {})
        origin={key:value for key,value in actual.items() if key.startswith('launch_origin_')}
        if name=='erc_mission_manager':
            assert origin==dict(launch_origin_monotonic_seconds=90.,
                launch_origin_utc='2026-09-11T00:00:00+00:00',launch_origin_basis='test_external_origin')
        else:assert not origin
        resolved[name]=actual
    assert not actions[4].additional_env  # RViz remains outside the numeric override.
    perception,navigation,manipulation,mission=(resolved[name] for name in names)
    assert perception['table_scene_enabled'] is selected
    assert manipulation['table_scene_required'] is selected
    assert manipulation['scene_cartesian_wall_seconds']==(1800. if selected else 420.)
    assert manipulation['delivery_evidence_enabled'] is selected
    assert mission['delivery_evidence_enabled'] is selected
    assert manipulation['stock_gripper_close_diagnostic_enabled'] is selected
    assert manipulation['stock_gripper_close_target_position']==(.0182 if selected else 0.)
    assert manipulation.get('place_finish_keep_torso_height_enabled', False) is selected
    assert manipulation.get('empty_pickup_parallel_geometry_enabled', False) is selected
    assert manipulation.get('settled_place_torso_retry_enabled', False) is selected
    assert manipulation.get('bin_clearance_timing_enabled', False) is False
    assert manipulation.get('withdrawal_half_timing_enabled', False) is selected
    assert manipulation.get('withdrawal_quarter_timing_enabled', False) is selected
    assert manipulation.get('place_finish_at_release_enabled', False) is selected
    assert manipulation.get('release_only_place_planning_enabled', False) is selected
    assert manipulation.get('release_only_clearance_timing_enabled', False) is selected
    assert mission.get('place_finish_at_release_enabled', False) is selected
    assert mission.get('mission_raw_contacts_enabled', False) is selected
    assert manipulation.get('empty_torso_planning_overlap_enabled', False) is selected
    assert manipulation.get('empty_head_timing_enabled', False) is False
    assert mission.get('empty_head_timing_enabled', False) is False
    assert navigation['max_angular_speed'] == (.70 if selected else .55)
    assert navigation.get('normal_translation_gain', .85) == (1.8 if selected else .85)
    assert navigation.get('normal_yaw_gain', 1.35) == (1.8 if selected else 1.35)
    assert manipulation['place_torso_height']==(.35 if selected else .30)
    assert manipulation['book_centered_place_clearance_height_m']==(.50 if selected else .40)
    assert manipulation['book_centered_place_height_above_point_m']==.27
    assert mission['bin_standoff_distance']==(.82 if selected else .72)
    assert 'bin_standoff_distance' not in navigation




def _expand_without_starting(module, configurations):
    context = launch.LaunchContext()
    context.launch_configurations.update(configurations)
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    actions = module._launch_nodes(context)
    assert len(actions) == 5 and len(actions[3].actions) == 1
    nodes = [*actions[:3], actions[3].actions[0]]
    names = ('erc_perception', 'erc_navigation', 'erc_manipulation', 'erc_mission_manager')
    resolved = {}
    for node, name in zip(nodes, names):
        node._perform_substitutions(context)
        resolved[name] = _scoped_parameters(node._Node__expanded_parameter_arguments, name)[0]
    return context, actions, resolved


@pytest.mark.parametrize('column,colour', [('1','red'),('2','blue'),('4','green'),('5','yellow')])
def test_two_required_arguments_select_all_installed_settings(monkeypatch, tmp_path, column, colour):
    module = _module(monkeypatch, tmp_path)
    context, actions, actual = _expand_without_starting(module, dict(
        shelf_column_number=column, book_colour=colour))
    assert context.launch_configurations['profile'] == 'collision_quality'
    assert context.launch_configurations['trial_timeout_seconds'] == '1800.0'
    selected = yaml.safe_load((PACKAGE/'config/collision_quality.yaml').read_bytes())
    for name, parameters in actual.items():
        assert parameters['shelf_column_number'] == int(column)
        assert parameters['book_colour'] == colour
        assert parameters['trial_timeout_seconds'] == 1800.0
        assert parameters['dry_run'] is False and parameters['team_name'] == 'TEAM_NAME'
        for key, expected in selected[name]['ros__parameters'].items():
            assert parameters[key] == expected and type(parameters[key]) is type(expected)
    # The launch delay and four ordinary process roles remain intact.
    assert actions[3].period == 2.0
    assert actual['erc_manipulation']['additional_arm_time_scale'] == 2.0
    assert actual['erc_manipulation']['settled_place_torso_skip_enabled'] is True
    assert actual['erc_manipulation']['settled_place_torso_retry_enabled'] is True
    assert actual['erc_manipulation']['pickup_parallel_geometry_enabled'] is True
    assert actual['erc_manipulation']['empty_pickup_parallel_geometry_enabled'] is True
    assert actual['erc_manipulation']['geometry_process_workers'] == 8
    assert actual['erc_manipulation']['pickup_post_retreat_parallel_geometry_enabled'] is True
    assert actual['erc_manipulation']['empty_pickup_setup_retiming_enabled'] is True
    assert actual['erc_manipulation']['place_carried_volume_parallel_enabled'] is True
    assert actual['erc_manipulation']['place_finish_keep_torso_height_enabled'] is True
    assert actual['erc_navigation']['max_angular_speed'] == .70
    assert actual['erc_navigation']['normal_translation_gain'] == 1.8
    assert actual['erc_navigation']['normal_yaw_gain'] == 1.8
    assert actual['erc_manipulation']['head_return_overlap_enabled'] is False
    assert actual['erc_mission_manager']['head_return_overlap_enabled'] is False
    assert actual['erc_manipulation']['loaded_place_speed_scale_cap'] == 2.0
    assert actual['erc_manipulation']['initial_stow_serial_timing_enabled'] is True
    assert actual['erc_manipulation']['place_transition_stop_enabled'] is True
    assert actual['erc_manipulation']['release_only_clearance_timing_enabled'] is True
    assert actual['erc_manipulation']['bin_clearance_timing_enabled'] is False
    assert actual['erc_manipulation']['withdrawal_half_timing_enabled'] is True
    assert actual['erc_manipulation']['withdrawal_quarter_timing_enabled'] is True
    assert actual['erc_manipulation']['place_finish_at_release_enabled'] is True
    assert actual['erc_manipulation']['release_only_place_planning_enabled'] is True
    assert actual['erc_mission_manager'].get('marker_search_first_probe_enabled', False) is True
    assert actual['erc_manipulation'].get('completed_head_hold_recheck_enabled', False) is True
    assert actual['erc_manipulation'].get('compact_extension_half_timing_enabled', False) is True
    assert actual['erc_mission_manager']['place_finish_at_release_enabled'] is True
    assert actual['erc_manipulation']['empty_torso_planning_overlap_enabled'] is True
    assert actual['erc_manipulation']['empty_head_timing_enabled'] is False
    assert actual['erc_mission_manager']['empty_head_timing_enabled'] is False
    assert actual['erc_manipulation']['placement_transport_speed_scale'] == 3.0
    assert actual['erc_manipulation']['place_parallel_geometry_enabled'] is True


def test_two_argument_parameters_match_explicit_selected_launch(monkeypatch, tmp_path):
    module = _module(monkeypatch, tmp_path)
    given = dict(shelf_column_number='2', book_colour='red')
    _, _, implicit = _expand_without_starting(module, given)
    _, _, explicit = _expand_without_starting(module, dict(given,
        profile='collision_quality', trial_timeout_seconds='1800', team_name='TEAM_NAME'))
    assert implicit == explicit


def test_explicit_base_profile_and_previous_timeout_remain_available(monkeypatch, tmp_path):
    module = _module(monkeypatch, tmp_path)
    context, actions, actual = _expand_without_starting(module, dict(
        shelf_column_number='2', book_colour='red', profile='default', trial_timeout_seconds='270'))
    assert context.launch_configurations['profile'] == 'default'
    for parameters in actual.values():
        assert parameters['trial_timeout_seconds'] == 270.0
    assert actual['erc_manipulation']['stock_gripper_close_target_position'] == 0.0
    assert 'raw_contacts_enabled' not in actual['erc_manipulation']
    for node in [*actions[:3], actions[3].actions[0]]:
        assert not node.additional_env


def test_missing_installed_default_profile_fails_before_node_creation(monkeypatch, tmp_path):
    module = _module(monkeypatch, tmp_path)
    monkeypatch.setattr(module, 'get_package_share_directory', lambda _: str(tmp_path))
    def forbidden_node(*args, **kwargs):
        raise AssertionError('Missing selected configuration must reject before creating node actions')
    monkeypatch.setattr(module, 'Node', forbidden_node)
    context = launch.LaunchContext()
    context.launch_configurations.update(dict(shelf_column_number='2', book_colour='red'))
    with pytest.raises(RuntimeError, match='profile is not installed'):
        module._launch_nodes(context)


def test_unknown_profile_rejected_before_node_expansion(monkeypatch,tmp_path):
    module=_module(monkeypatch,tmp_path)
    context=launch.LaunchContext()
    context.launch_configurations.update(dict(shelf_column_number='2',book_colour='red',profile='typo'))
    with pytest.raises(RuntimeError,match='Unknown profile'):
        module._launch_nodes(context)
