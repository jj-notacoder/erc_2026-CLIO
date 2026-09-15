"""Actual serial sender and controlled post-torso feedback; no plant model."""
import math
import pytest
from test_initial_stow_serial_timing import fixture, run, start, serial


def residual_fixture(monkeypatch, *, error=.003, velocity=.0015):
    n, clock, events = fixture(monkeypatch)
    torso = serial.IK_JOINTS[0]
    original_send = n.torso_client.send_goal_async
    n.torso_client.update_endpoint = False

    def send(goal):
        value = original_send(goal)
        n.joints[torso] = serial.HOME[0] + error
        n._joint_velocities[torso] = velocity
        return value

    n.torso_client.send_goal_async = send
    return n, clock, events


@pytest.mark.parametrize('residual', ['position', 'velocity', 'both'])
def test_original_torso_completion_waits_for_residual_to_settle(monkeypatch, residual):
    n, clock, events = residual_fixture(monkeypatch,
        error=.003 if residual != 'velocity' else 0.,
        velocity=.0015 if residual != 'position' else 0.)
    update = clock.hook
    settled = []

    def progress():
        update()
        if clock.ns >= 1_080_000_000:
            n.joints[serial.IK_JOINTS[0]] = serial.HOME[0]
            n._joint_velocities[serial.IK_JOINTS[0]] = 0.
            if not settled: settled.append(clock.ns)

    clock.hook = progress
    assert run(n) is True
    sends = [e for e in events if e[0] == 'send']
    assert [e[1] for e in sends] == ['torso', 'left', 'right']
    assert sends[1][2] - settled[0] >= 100_000_000
    assert n.torso_client.goals[0].trajectory.points[0].time_from_start.sec == 2
    assert n.torso_client.goals[0].trajectory.points[0].time_from_start.nanosec == 500_000_000
    assert not n._goal_handles and not n._pending_retained_acceptances
    assert n._initial_stow_serial_owner is None and n._right_parked
    assert not any(e[:2] == ('status', 'initial_stow_torso_settle_refused') for e in events)


@pytest.mark.parametrize('residual', ['position', 'velocity', 'both'])
def test_never_settled_torso_refuses_with_exact_measured_diagnostic(monkeypatch, residual):
    error = .003 if residual != 'velocity' else 0.
    velocity = .0015 if residual != 'position' else 0.
    n, clock, events = residual_fixture(monkeypatch, error=error, velocity=velocity)
    with pytest.raises(TimeoutError, match='measured stop ROS deadline'):
        run(n)
    assert 10.75 < clock.wall < 10.80
    assert not n.arm_client.goals and not n.right_arm_client.goals
    assert n._cancel.is_set() and n._initial_stow_serial_owner is not None
    diagnostics = [e[2] for e in events if e[:2] == ('status', 'initial_stow_torso_settle_refused')]
    d, = diagnostics
    sample = d['last_valid_sample']
    assert sample['q'] == serial.HOME[0] + error and sample['v'] == velocity
    assert sample['target'] == serial.HOME[0]
    assert sample['position_within'] is (residual == 'velocity')
    assert sample['velocity_within'] is (residual == 'position')
    assert d['completion_ros_ns'] == 1_000_000_000
    assert d['position_tolerance_m'] == .002 and d['velocity_limit_mps'] == .001
    assert d['ros_budget_seconds'] == .75 and d['wall_budget_seconds'] == 3.
    assert d['required_producer_span_ns'] == 100_000_000


def test_torso_residual_reappearing_resets_the_consecutive_producer_span(monkeypatch):
    n, clock, events = residual_fixture(monkeypatch, error=0., velocity=0.)
    update = clock.hook

    def progress():
        update()
        n.joints[serial.IK_JOINTS[0]] = serial.HOME[0] + (.003 if clock.ns == 1_080_000_000 else 0.)

    clock.hook = progress
    assert run(n)
    left = next(e for e in events if e[:2] == ('send', 'left'))
    assert left[2] >= 1_200_000_000


@pytest.mark.parametrize('producer', ['joints', 'odom', 'both'])
def test_precompletion_producer_samples_cannot_qualify_initial_stop(monkeypatch, producer):
    n, clock, events = residual_fixture(monkeypatch, error=0., velocity=0.)
    update = clock.hook

    def progress():
        update()
        if clock.ns <= 1_080_000_000:
            if producer in ('joints', 'both'):
                n._joint_stamps_ns.update(dict.fromkeys(serial.NAMES, 1_000_000_000))
            if producer in ('odom', 'both'):
                n._staging_odom['stamp_ns'] = 1_000_000_000

    clock.hook = progress
    assert run(n)
    left = next(e for e in events if e[:2] == ('send', 'left'))
    assert left[2] >= 1_200_000_000


@pytest.mark.parametrize('fault', ['stale', 'future', 'reversed_joint', 'reversed_odom',
    'clock_rollback', 'nan', 'moving_base', 'cancel', 'cancel_clear', 'owner', 'wire'])
def test_fatal_feedback_or_ownership_faults_are_not_torso_settling(monkeypatch, fault):
    n, clock, events = residual_fixture(monkeypatch)
    update = clock.hook

    def progress():
        update()
        if clock.ns < 1_040_000_000: return
        name = serial.IK_JOINTS[0]
        if fault == 'stale': n._joint_stamps_ns[name] = clock.ns - 150_000_001
        if fault == 'future': n._joint_stamps_ns[name] = clock.ns + 1
        if fault == 'reversed_joint': n._joint_stamps_ns[name] = 1_019_999_999
        if fault == 'reversed_odom': n._staging_odom['stamp_ns'] = 1_019_999_999
        if fault == 'clock_rollback': clock.ns = 999_999_999
        if fault == 'nan': n.joints[name] = math.nan
        if fault == 'moving_base': n._staging_odom['linear_speed'] = .005001
        if fault == 'cancel': n._cancel.set()
        if fault == 'cancel_clear': n._cancel.set(); n._cancel.clear()
        if fault == 'owner': n._initial_stow_serial_owner = object()
        if fault == 'wire': n._raw_contacts_first_failure = 'wire_during_torso_settle'

    clock.hook = progress
    with pytest.raises(RuntimeError): run(n)
    assert clock.wall < 10.1 and n._cancel.is_set()
    assert not n.arm_client.goals and not n.right_arm_client.goals
    d, = [e[2] for e in events if e[:2] == ('status', 'initial_stow_torso_settle_refused')]
    assert d['last_valid_sample']['sample_ros_ns'] == 1_020_000_000
    assert len(d['reason']) <= 256


def test_stalled_ros_clock_retains_original_wall_deadline(monkeypatch):
    n, clock, events = residual_fixture(monkeypatch)
    clock.sleep = lambda seconds: setattr(clock, 'wall', clock.wall + seconds)
    with pytest.raises(TimeoutError, match='measured stop wall deadline'): run(n)
    assert 13. <= clock.wall < 13.1
    assert not n.arm_client.goals and not n.right_arm_client.goals
    assert len([e for e in events if e[:2] == ('status', 'initial_stow_torso_settle_refused')]) == 1


@pytest.mark.parametrize('change', ['position', 'velocity'])
def test_strict_torso_gate_is_rechecked_at_original_arm_publication(monkeypatch, change):
    n, _, _ = residual_fixture(monkeypatch, error=0., velocity=0.)

    def wait(**unused):
        if change == 'position': n.joints[serial.IK_JOINTS[0]] += .002001
        else: n._joint_velocities[serial.IK_JOINTS[0]] = .001001
        return True

    n.arm_client.wait_for_server = wait
    with pytest.raises(RuntimeError, match='torso not at original stopped HOME'): run(n)
    assert not n.arm_client.goals and not n.right_arm_client.goals


def test_diagnostic_publication_failure_does_not_replace_settling_fault(monkeypatch):
    n, _, _ = residual_fixture(monkeypatch)
    n._publish_status = lambda *a, **k: (_ for _ in ()).throw(ValueError('diagnostic unavailable'))
    with pytest.raises(TimeoutError, match='measured stop ROS deadline'): run(n)
    assert n._cancel.is_set() and n._initial_stow_serial_owner is not None
    assert not n.arm_client.goals and not n.right_arm_client.goals


@pytest.mark.parametrize('phase', ['before_completion', 'after_start', 'after_send'])
def test_torso_settling_exception_cannot_be_used_outside_initial_phase(monkeypatch, phase):
    n, clock, _ = fixture(monkeypatch)
    owner = start(n)
    if phase != 'before_completion': owner.torso_completed_ns = clock.ns
    if phase == 'after_start': owner.start = owner.check()
    if phase == 'after_send': owner.sent = True
    n.joints[serial.IK_JOINTS[0]] += .003
    with n._lock:
        with pytest.raises(RuntimeError, match='phase identity changed'):
            owner.snapshot_locked(torso_after=clock.ns)
    assert not n.arm_client.goals and not n.right_arm_client.goals


@pytest.mark.parametrize('residual', ['position', 'velocity', 'arm_velocity'])
def test_torso_only_extended_window_can_admit_late_measured_stop(monkeypatch, residual):
    n, clock, events = residual_fixture(monkeypatch,
        error=.003 if residual == 'position' else 0.,
        velocity=.0015 if residual == 'velocity' else 0.)
    if residual == 'arm_velocity': n._joint_velocities[serial.ARM_NAMES[0]] = .02
    update = clock.hook

    def progress():
        update()
        if clock.ns >= 1_600_000_000:
            n.joints[serial.IK_JOINTS[0]] = serial.HOME[0]
            n._joint_velocities[serial.IK_JOINTS[0]] = 0.
            n._joint_velocities[serial.ARM_NAMES[0]] = 0.

    clock.hook = progress
    assert run(n) is True
    sends = [e for e in events if e[0] == 'send']
    assert [e[1] for e in sends] == ['torso', 'left', 'right']
    assert sends[1][2] == 1_700_000_000
    assert not n._goal_handles and not n._pending_retained_acceptances
    assert n._initial_stow_serial_owner is None and n._right_parked


@pytest.mark.parametrize('beyond_ns', [0, 1])
def test_torso_deadline_equality_passes_but_one_nanosecond_beyond_refuses(monkeypatch, beyond_ns):
    n, clock, _ = fixture(monkeypatch)
    owner = start(n); owner.torso_completed_ns = clock.ns
    update = clock.hook
    n.joints[serial.IK_JOINTS[0]] = serial.HOME[0] + .003

    def sleep(seconds):
        clock.wall += .05
        clock.ns += 50_000_000
        if clock.ns == 1_750_000_000: clock.ns += beyond_ns
        update()
        if clock.ns >= 1_650_000_000: n.joints[serial.IK_JOINTS[0]] = serial.HOME[0]

    clock.sleep = sleep
    if beyond_ns:
        with pytest.raises(TimeoutError, match='measured stop ROS deadline'):
            owner.wait_stopped(torso_after=1_000_000_000)
        assert owner.torso_settle_diagnostic['wait_elapsed_ros_ns'] == 750_000_001
    else:
        snap = owner.wait_stopped(torso_after=1_000_000_000)
        assert snap['now'] == 1_750_000_000 and not owner.torso_settle_active
    assert not n.torso_client.goals and not n.arm_client.goals and not n.right_arm_client.goals


@pytest.mark.parametrize('phase,budget_ns', [('before_arm', 500_000_000), ('after_arm', 2_000_000_000)])
def test_non_torso_stop_phases_keep_their_original_ros_budgets(monkeypatch, phase, budget_ns):
    n, clock, _ = fixture(monkeypatch);n.timeout = 4.
    owner = start(n);n._joint_velocities[serial.ARM_NAMES[0]] = .02
    kwargs = {} if phase == 'before_arm' else {'after': clock.ns}
    with pytest.raises(TimeoutError, match='measured stop ROS deadline'):
        owner.wait_stopped(**kwargs)
    assert budget_ns < clock.ns - 1_000_000_000 <= budget_ns + 20_000_000
    assert not owner.torso_settle_active and owner.torso_settle_diagnostic is None
    assert not n.arm_client.goals and not n.right_arm_client.goals


@pytest.mark.parametrize('fault', ['stale', 'future', 'reversed_joint', 'reversed_odom', 'cancel', 'cancel_clear'])
def test_extended_torso_wait_retains_freshness_and_cancellation_after_old_deadline(monkeypatch, fault):
    n, clock, events = residual_fixture(monkeypatch)
    update = clock.hook

    def progress():
        update()
        if clock.ns < 1_620_000_000: return
        name = serial.IK_JOINTS[0]
        if fault == 'stale': n._joint_stamps_ns[name] = clock.ns - 150_000_001
        if fault == 'future': n._joint_stamps_ns[name] = clock.ns + 1
        if fault == 'reversed_joint': n._joint_stamps_ns[name] = 1_599_999_999
        if fault == 'reversed_odom': n._staging_odom['stamp_ns'] = 1_599_999_999
        if fault == 'cancel': n._cancel.set()
        if fault == 'cancel_clear': n._cancel.set(); n._cancel.clear()

    clock.hook = progress
    with pytest.raises(RuntimeError): run(n)
    assert clock.ns == 1_620_000_000 and n._cancel.is_set()
    assert not n.arm_client.goals and not n.right_arm_client.goals
    d, = [e[2] for e in events if e[:2] == ('status', 'initial_stow_torso_settle_refused')]
    assert d['last_valid_sample']['sample_ros_ns'] == 1_600_000_000


@pytest.mark.parametrize('blocker', ['moving_arm', 'late_first_good'])
def test_refusal_diagnostic_distinguishes_arm_motion_from_short_producer_window(monkeypatch, blocker):
    import json
    n, clock, events = residual_fixture(monkeypatch, error=0., velocity=0.)
    update = clock.hook
    if blocker == 'moving_arm': n._joint_velocities[serial.ARM_NAMES[0]] = .02
    else: n.joints[serial.IK_JOINTS[0]] = serial.HOME[0] + .003

    def progress():
        update()
        if blocker == 'late_first_good':
            n.joints[serial.IK_JOINTS[0]] = serial.HOME[0] + (.003 if clock.ns < 1_680_000_000 else 0.)

    # Set the residual after the original torso action has completed too.
    if blocker == 'late_first_good':
        original_send = n.torso_client.send_goal_async
        def send(goal):
            future = original_send(goal)
            n.joints[serial.IK_JOINTS[0]] = serial.HOME[0] + .003
            return future
        n.torso_client.send_goal_async = send
    clock.hook = progress
    with pytest.raises(TimeoutError, match='measured stop ROS deadline'): run(n)
    d, = [e[2] for e in events if e[:2] == ('status', 'initial_stow_torso_settle_refused')]
    s = d['last_valid_sample']
    assert s['position_within'] is s['velocity_within'] is True
    assert s['arm_stop_velocity_limit'] == .01 and s['wait_elapsed_ros_ns'] == 760_000_000
    if blocker == 'moving_arm':
        assert s['arm_max_abs_velocity'] == .02 and s['arms_within_stop_limit'] is False
        assert s['first_good_ros_ns'] is s['minimum_joint_producer_span_ns'] is s['odom_producer_span_ns'] is None
    else:
        assert s['arm_max_abs_velocity'] == 0. and s['arms_within_stop_limit'] is True
        assert s['first_good_ros_ns'] == 1_680_000_000
        assert s['minimum_joint_producer_span_ns'] == s['odom_producer_span_ns'] == 80_000_000
    assert json.loads(json.dumps(d)) == d
    assert not n.arm_client.goals and not n.right_arm_client.goals


def use_slow_simulation(n, clock, *, rate=.1):
    ros_clock = n.get_clock()
    ros_clock.ros_time_is_active = True
    n.get_clock = lambda: ros_clock
    n.timeout = 30.

    def sleep(seconds):
        clock.wall += seconds
        clock.ns += round(seconds * rate * 1e9)
        clock.hook()

    clock.sleep = sleep


@pytest.mark.parametrize('rate', [.13, .05])
def test_slow_gazebo_completes_serial_stow_after_measured_torso_settle(monkeypatch, rate):
    n, clock, events = residual_fixture(monkeypatch)
    use_slow_simulation(n, clock, rate=rate)
    update = clock.hook
    settled = []

    def progress():
        update()
        if clock.ns >= 1_600_000_000:
            n.joints[serial.IK_JOINTS[0]] = serial.HOME[0]
            n._joint_velocities[serial.IK_JOINTS[0]] = 0.
            if not settled: settled.append(clock.ns)

    clock.hook = progress
    assert run(n)
    assert clock.wall > 13.  # The former three-second wall deadline would abort.
    sends = [e for e in events if e[0] == 'send']
    assert [e[1] for e in sends] == ['torso', 'left', 'right']
    assert sends[1][2] - settled[0] >= 100_000_000
    assert n._right_parked and n._initial_stow_serial_owner is None
    assert not n._goal_handles and not n._pending_retained_acceptances


@pytest.mark.parametrize('phase', ['before_arm', 'after_arm'])
def test_slow_simulation_arm_stop_keeps_measured_endpoint_and_producer_span(monkeypatch, phase):
    n, clock, _ = fixture(monkeypatch)
    use_slow_simulation(n, clock)
    owner = start(n)
    n._joint_velocities[serial.ARM_NAMES[0]] = .02
    settled_ns = 1_400_000_000 if phase == 'before_arm' else 2_200_000_000
    update = clock.hook

    def progress():
        update()
        if clock.ns >= settled_ns:
            n._joint_velocities[serial.ARM_NAMES[0]] = 0.
            n.joints.update(zip(serial.ARM_JOINTS, serial.HOME[1:]))

    clock.hook = progress
    snap = owner.wait_stopped(**({} if phase == 'before_arm' else {'after': clock.ns}))
    assert snap['now'] >= settled_ns + 100_000_000
    assert clock.wall > (13. if phase == 'before_arm' else 18.)
    assert not n.arm_client.goals and not n.right_arm_client.goals


def test_slow_simulation_unsettled_torso_still_hits_same_ros_deadline(monkeypatch):
    n, clock, events = residual_fixture(monkeypatch)
    use_slow_simulation(n, clock)
    with pytest.raises(TimeoutError, match='measured stop ROS deadline'):
        run(n)
    assert 750_000_000 < clock.ns - 1_000_000_000 <= 752_000_000
    assert 17.5 < clock.wall < 17.55
    assert not n.arm_client.goals and not n.right_arm_client.goals
    diagnostic, = [e[2] for e in events if e[:2] == ('status', 'initial_stow_torso_settle_refused')]
    assert diagnostic['ros_budget_seconds'] == .75
    assert diagnostic['wall_budget_seconds'] == 30.
    assert diagnostic['wall_progress_timeout_seconds'] == 3.


@pytest.mark.parametrize('frozen', ['clock', 'joints', 'odom'])
def test_simulation_stalled_clock_or_producer_keeps_short_wall_watchdog(monkeypatch, frozen):
    n, clock, _ = residual_fixture(monkeypatch)
    use_slow_simulation(n, clock, rate=.001)
    update = clock.hook
    stamps = dict(n._joint_stamps_ns)

    def progress():
        update()
        if frozen == 'joints': n._joint_stamps_ns.update(stamps)
        if frozen == 'odom': n._staging_odom['stamp_ns'] = 1_000_000_000

    clock.hook = progress
    if frozen == 'clock':
        clock.sleep = lambda seconds: setattr(clock, 'wall', clock.wall + seconds)
    with pytest.raises(TimeoutError, match='measured stop wall deadline'):
        run(n)
    assert 13. <= clock.wall < 13.1
    assert not n.arm_client.goals and not n.right_arm_client.goals
    assert n._cancel.is_set() and n._initial_stow_serial_owner is not None


def test_simulation_continuous_slow_progress_is_bounded_by_command_timeout(monkeypatch):
    n, clock, _ = residual_fixture(monkeypatch)
    use_slow_simulation(n, clock, rate=.001)
    n.timeout = 4.
    with pytest.raises(TimeoutError, match='measured stop wall deadline'):
        run(n)
    assert 14. <= clock.wall < 14.1
    assert not n.arm_client.goals and not n.right_arm_client.goals


@pytest.mark.parametrize('fault', ['cancel', 'stale', 'future', 'nan', 'moving_base'])
def test_slow_simulation_fault_after_old_wall_deadline_still_refuses(monkeypatch, fault):
    n, clock, _ = residual_fixture(monkeypatch)
    use_slow_simulation(n, clock)
    update = clock.hook

    def progress():
        update()
        if clock.wall < 14.: return
        name = serial.IK_JOINTS[0]
        if fault == 'cancel': n._cancel.set()
        if fault == 'stale': n._joint_stamps_ns[name] = clock.ns - 150_000_001
        if fault == 'future': n._joint_stamps_ns[name] = clock.ns + 1
        if fault == 'nan': n.joints[name] = math.nan
        if fault == 'moving_base': n._staging_odom['linear_speed'] = .005001

    clock.hook = progress
    with pytest.raises(RuntimeError):
        run(n)
    assert 14. <= clock.wall < 14.1
    assert not n.arm_client.goals and not n.right_arm_client.goals
