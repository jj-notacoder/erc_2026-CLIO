"""Micro-only timing with real ROS messages/evaluators and fake clocks/publishers."""
import ast
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip('rclpy')
from erc_phase1_solution import fine_gripper_close as fine
from erc_phase1_solution import manipulation_node as manipulation
from test_fine_gripper_close import fixture, pressure_fixture, set_feedback
from test_fine_micro_settling_profile import acquisition_fixture, candidate
from test_fine_pressure_resettle import records
from test_contact_force_timing import BOOK, LEFT, RIGHT, force_message


def selected(**kwargs):
    return candidate(micro_motion_seconds=.40, maximum_force=30., **kwargs)


def test_micro_duration_is_opt_in_and_does_not_change_any_other_limit():
    baseline = fine.FineGripLimits()
    changed = replace(baseline, micro_motion_seconds=.40)
    assert baseline.micro_motion_seconds == baseline.motion_seconds == .20
    assert {name for name, value in asdict(baseline).items()
            if asdict(changed)[name] != value} == {'micro_motion_seconds'}
    assert changed.dwell_seconds == .20 and changed.motion_seconds == .20
    assert changed.step_wall_seconds == 5. and changed.total_wall_seconds == 300.
    assert changed.micro_preload_steps == 20 and changed.micro_preload_limit == 2e-6


@pytest.mark.parametrize('duration', [.20, .40, 1.0])
def test_duration_boundary_values_are_accepted(duration):
    assert fine.FineGripLimits(micro_motion_seconds=duration).micro_motion_seconds == duration


@pytest.mark.parametrize('duration', [0., -.2, .199999, 1.000001, float('nan'), float('inf')])
def test_invalid_duration_rejects_before_commands(duration):
    with pytest.raises(ValueError):
        fine.FineGripLimits(micro_motion_seconds=duration)


@pytest.mark.parametrize('duration', [None, .40, .199999, 1.000001, float('nan')])
def test_actual_parameter_declaration_and_constructor_assignment_use_new_field(duration):
    # Execute the actual declaration method and the unchanged constructor's
    # complete limits assignment without starting a ROS node or loading meshes.
    values = {}
    node = SimpleNamespace(declare_parameter=lambda name, default: values.setdefault(name, default),
                           adaptive_contact_force_maximum=8.)
    manipulation.ManipulationNode._declare_parameters(node)
    assert values['fine_gripper_micro_motion_seconds'] == .20
    config = yaml.safe_load((Path(manipulation.__file__).parents[1]/'config/solution.yaml').read_text())
    assert config['erc_manipulation']['ros__parameters']['fine_gripper_micro_motion_seconds'] == .20
    if duration is not None:
        values['fine_gripper_micro_motion_seconds'] = duration
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    module = ast.parse(Path(manipulation.__file__).read_text())
    cls = next(x for x in module.body if isinstance(x, ast.ClassDef) and x.name == 'ManipulationNode')
    init = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '__init__')
    statement = next(x for x in init.body if isinstance(x, ast.Assign)
                     and any(isinstance(t, ast.Attribute) and t.attr == 'fine_gripper_limits' for t in x.targets))
    code = compile(ast.Module(body=[statement], type_ignores=[]), '<actual fine-limit constructor mapping>', 'exec')
    invalid = duration is not None and (duration < .20 or duration > 1. or duration != duration)
    if invalid:
        with pytest.raises(ValueError): exec(code, {'self': node, 'FineGripLimits': fine.FineGripLimits})
    else:
        exec(code, {'self': node, 'FineGripLimits': fine.FineGripLimits})
        assert node.fine_gripper_limits.micro_motion_seconds == (.20 if duration is None else duration)
        assert node.fine_gripper_limits.motion_seconds == .20


def test_real_seek_and_micro_publications_keep_phase_durations_and_full_postmotion_dwell(monkeypatch):
    c, node, now, clock, messages, statuses = fixture(monkeypatch, position=.069, limits=selected())
    touching = []
    publications = []
    publish = node.gripper_pub.publish
    def capture(message):
        publications.append((now.nanoseconds, message, c._fine_phase))
        publish(message)
    node.gripper_pub.publish = capture
    def sensors():
        set_feedback(node, now, c._fine_command)
        if c._fine_phase == 'fine' and not touching:
            touching.append(now.nanoseconds)
            node._on_contacts(force_message(now.nanoseconds, (LEFT, BOOK, 3.), (RIGHT, BOOK, 3.)))
        elif touching:
            # Strong pressure during declared micro motion still cannot acquire.
            after_step = c._fine_micro_active and c._fine_command < c._fine_micro_reference-50e-9
            pairs = ((LEFT, BOOK, 8.5), (RIGHT, BOOK, 8.7)) if after_step else ((RIGHT, BOOK, .3),)
            node._on_contacts(force_message(now.nanoseconds, *pairs))
            if after_step and now.nanoseconds < c._fine_motion_end_ns+200_000_000:
                assert c._fine_result is None and not node._transport_lock_engaged
    clock.on_sleep = sensors
    assert c.run()[0] is True and c._fine_fault is None
    commands = records(statuses, 'public_position_command')
    assert len(commands) == len(publications) == len(messages)
    for event, (_, message, phase) in zip(commands, publications):
        stamp = message.points[0].time_from_start
        assert stamp.sec+stamp.nanosec/1e9 == pytest.approx(event['motion_seconds'])
        assert event['phase'] == phase
    assert {x['motion_seconds'] for x in commands if x['phase'] == 'preclose'} == {1.0}
    assert {x['motion_seconds'] for x in commands if x['phase'] == 'coarse'} == {.32}
    assert {x['motion_seconds'] for x in commands if x['phase'] == 'fine' and x['purpose'] == 'close'} == {.20}
    assert {x['motion_seconds'] for x in commands if x['purpose'] == 'first_bilateral_hold'} == {.02}
    micro = [(event, publication) for event, publication in zip(commands, publications)
             if event['phase'] == 'micro_preload' and event['purpose'] == 'close']
    assert len(micro) == 1 and micro[0][0]['motion_seconds'] == .40
    end = micro[0][1][0]+400_000_000
    endpoint = records(statuses, 'stationary_endpoint')[-1]
    assert endpoint['stationarity_diagnostic']['command_motion_end_ros_ns'] == end
    assert endpoint['stationary_start_ros_ns'] >= end
    assert endpoint['stationarity_diagnostic']['longest_confirmed_stable_interval_seconds'] >= .20
    pressure = records(statuses, 'pressure_measurement')[-1]
    assert pressure['minimum_force'] == 8. and pressure['pressure_evidence']['verified']
    assert all(pressure['pressure_windows'][side]['span_seconds'] >= .05 for side in ('left', 'right'))
    assert records(statuses, 'micro_preload_started')[-1]['motion_seconds'] == .40
    assert c._fine_wall_deadline == 900.


@pytest.mark.parametrize('fault', ['force', 'cancel'])
def test_live_fault_in_second_half_of_longer_micro_move_holds_without_later_close(monkeypatch, fault):
    c, node, now, clock, messages, statuses = acquisition_fixture(monkeypatch)
    c.limits = selected()
    old = clock.on_sleep
    fired = []
    def sensors():
        old()
        if (not fired and c._fine_micro_active and c._fine_command < c._fine_micro_reference-50e-9
                and c._fine_motion_end_ns-150_000_000 <= now.nanoseconds < c._fine_motion_end_ns):
            fired.append(now.nanoseconds)
            if fault == 'force': node._on_contacts(force_message(now.nanoseconds, (RIGHT, BOOK, 30.1)))
            else: node._cancel.set()
    clock.on_sleep = sensors
    assert c.run()[0] is False and fired
    assert c._fine_fault == ('force_overload' if fault == 'force' else 'cancelled')
    commands = [x for x in records(statuses, 'public_position_command') if x['purpose'] == 'close']
    assert len(commands) == 1 and commands[0]['motion_seconds'] == .40
    before = len(messages)
    with c._fine_lock:
        assert not c._publish_once_locked(c._fine_command-100e-9, .40, 'close')
    assert len(messages) == before and c._fine_result is None


@pytest.mark.parametrize('which', ['step', 'total'])
def test_longer_motion_spends_existing_deadline_and_cannot_shorten_dwell(monkeypatch, which):
    c, node, now, clock, messages, statuses = pressure_fixture(monkeypatch, limits=selected())
    target = c._fine_command-100e-9
    with c._fine_lock:
        assert c._publish_once_locked(target, c.limits.micro_motion_seconds, 'close')
    def sensors():
        set_feedback(node, now, target)
        c._fine_side_last = {'right': (now.nanoseconds, .3)}
    clock.on_sleep = sensors
    if which == 'total': c._fine_wall_deadline = .5
    assert not c._wait_motion_and_stationary(target, micro=True, wall_deadline=.5 if which == 'step' else 15.)
    assert clock.wall <= .5021 and c._fine_fault in ('step_wall_timeout', 'fine_grip_wall_timeout')
    assert not records(statuses, 'stationary_endpoint') and not records(statuses, 'pressure_measurement')
    assert c._fine_result is None
