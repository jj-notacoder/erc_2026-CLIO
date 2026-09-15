"""Additive integration/liveness models; retain the sealed 60 original cases."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import candidate_composition_support as support
from test_preopen_stationary import node, run, gate

ROOT=Path(__file__).resolve().parents[1]
NODE=ROOT/'erc_phase1_solution/manipulation_node.py'


@pytest.mark.parametrize('schedule',['steady','finite_race_burst','always_races'])
def test_500hz_ros_contact_schedule_preserves_span_or_fails_bounded(node,schedule):
    # Fixed 2 ms ROS / 5 ms wall ticks model .4 ROS/wall, not DDS throughput.
    def sleep(seconds):
        assert not node.command_lock._is_owned()
        node.require_sensor_available()
        node.wall+=.005;node.now+=2_000_000;node.ticks+=1
        node._contact_generation+=1
        node.feed()
    node.sleep=sleep
    gate.time.sleep=sleep
    def during_final():
        if schedule=='always_races' or (schedule=='finite_race_burst' and node.context_checks<=4):
            node._contact_generation+=1
    node.on_context=during_final
    if schedule=='always_races':
        with pytest.raises(gate.PreopenStationaryRejected,match='measurement_timeout'):run(node)
        assert node.wall<=10.01 and node.context_checks>4
        assert not any(f['verified'] for _,f in node.events)
    else:
        result=run(node)
        assert result['producer_stamp_ns']-result['stationary_start_ns']>=100_000_000
        assert node.ticks>=51 and node.wall<.35
        assert node.context_checks==(5 if schedule=='finite_race_burst' else 1)
    assert not node._delivery_measurement_active


def test_generation_retry_does_not_hide_concurrent_hazard(node):
    def disturb():
        node._contact_generation+=1
        node._payload_hazard_latched='concurrent_hazard'
    node.on_context=disturb
    with pytest.raises(gate.PreopenStationaryRejected,match='concurrent_hazard'):run(node)
    assert not any(f['verified'] for _,f in node.events)


@pytest.mark.parametrize('selected',[False,True])
def test_actual_parameter_declaration_and_constructor_assignment(selected):
    tree=ast.parse(NODE.read_text())
    owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ManipulationNode')
    declare=copy.deepcopy(next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='_declare_parameters'))
    init=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    assignment=copy.deepcopy(next(n for n in init.body if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Attribute) and t.attr=='preopen_stationary_enabled' for t in n.targets)))
    defaults={}
    fake=NS(declare_parameter=lambda n,v:defaults.__setitem__(n,v),
        get_parameter=lambda n:NS(value=selected))
    scope={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[declare],type_ignores=[])),'actual declaration','exec'),scope)
    scope['_declare_parameters'](fake)
    assert defaults['preopen_stationary_enabled'] is False
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),'actual constructor assignment','exec'),{'self':fake})
    assert fake.preopen_stationary_enabled is selected
    selected_config=(ROOT/'config/collision_quality.yaml').read_text()
    assert selected_config.count('    preopen_stationary_enabled: true\n')==1
    assert selected_config.count('    placement_transport_speed_scale: 3.0\n')==1


@pytest.mark.parametrize('damage',['missing','duplicate','unrelated'])
def test_preopen_inverse_rejects_undeclared_source(damage):
    source=NODE.read_text()
    record=json.loads((ROOT/'test/fixtures/preopen_stationary_inverse.json').read_text())
    insertion=record['insertions'][0]
    if damage=='missing':source=source.replace(insertion,'',1)
    elif damage=='duplicate':source+=insertion
    else:source+='\n# unrelated source edit\n'
    with pytest.raises(AssertionError):support.restore_preopen_stationary_source(source)


def test_existing_range_and_raw_full_source_assertions_remain_composed():
    source=NODE.read_text()
    assert hashlib.sha256(support.restore_preopen_stationary_source(source).encode()).hexdigest()=='b4ecd4049b23ab62362af7676d28c2f219e55f60fa2103df08d8bd364c52979e'
    assert hashlib.sha256(support.restore_extended_arm_speed_source(source).encode()).hexdigest()=='14df13cfb6d80574b3a5a292b2f422a977f343e677ab9f17a986bbc0a3f71366'
    assert hashlib.sha256(support.restore_raw_contacts_source(source).encode()).hexdigest()=='06bef5efb58b73301aced8e942a1df5528a931a3d2ce46e74d307fa29a5810bf'


@pytest.mark.parametrize('module',['node','timing'])
@pytest.mark.parametrize('damage',['missing','duplicate','unrelated'])
def test_three_speed_inverse_rejects_extra_or_missing_changes(module,damage):
    if module=='node':
        source=NODE.read_text()
        fragment='float(duration) / 3.0 <= command_duration'
        restore=support.restore_preopen_speed_source
        if damage=='missing':source=source.replace(fragment,fragment.replace('3.0','3.1'))
        elif damage=='duplicate':source+='\n# '+fragment+'\n'
        else:source+='\n# unrelated code\n'
    else:
        source=(ROOT/'erc_phase1_solution/arm_trajectory_timing.py').read_bytes()
        fragment=b'1.0 <= scale <= 3.0'
        restore=support.restore_preopen_speed_timing_bytes
        if damage=='missing':source=source.replace(fragment,fragment.replace(b'3.0',b'3.1'))
        elif damage=='duplicate':source+=b'\n# '+fragment+b'\n'
        else:source+=b'\n# unrelated code\n'
    with pytest.raises(AssertionError):restore(source)
