"""Independent full-schema ROS oracle; actual unchanged R52 state transitions."""
import math
import struct
import pytest

from raw_contacts_oracle_support import (CallbackPair, Contacts, IDLEncoder, canonical,
                            deserialize_message, fixture as f, full_message,
                            raw, serialize_message, stream_messages)

@pytest.mark.parametrize('seed', range(18))
def test_complete_installed_schema_matches_actual_ros_round_trip(seed):
    message = full_message(seed)
    data = serialize_message(message)
    stock = deserialize_message(data, Contacts)
    assert canonical(raw.decode_contacts_cdr(data)) == canonical(stock)
    assert canonical(stock) == canonical(message)

@pytest.mark.parametrize('endian', ['<', '>'])
@pytest.mark.parametrize('seed', [0, 1, 2, 7])
def test_opposite_endian_and_alignment_is_stock_checked(endian, seed):
    message = full_message(seed)
    data = IDLEncoder(endian).write(message)
    stock = deserialize_message(data, Contacts)
    assert canonical(stock) == canonical(message)
    assert canonical(raw.decode_contacts_cdr(data)) == canonical(stock)

@pytest.mark.parametrize('length', range(17))
def test_string_alignment_zero_through_sixteen_bytes(length):
    message = full_message(0, names=['x' * length])
    data = serialize_message(message)
    assert canonical(raw.decode_contacts_cdr(data)) == canonical(deserialize_message(data, Contacts))

def test_empty_contacts_remains_a_real_delivery():
    data = serialize_message(Contacts())
    seen = []
    node, _ = f.make_node()
    raw.raw_contacts_callback(node, seen.append)(data)
    assert len(seen) == 1 and canonical(seen[0]) == canonical(Contacts())

@pytest.mark.parametrize('text', ['\x00', 'left\x00right', 'µ\x00棚'])
def test_embedded_nul_matches_stock_string_conversion(text):
    message = full_message(0, names=[text])
    data = serialize_message(message)
    assert canonical(raw.decode_contacts_cdr(data)) == canonical(deserialize_message(data, Contacts))

def test_owned_immutable_input_views_and_message_isolation():
    data = bytearray(serialize_message(full_message(1)))
    decoded = raw.decode_contacts_cdr(data)
    before = canonical(decoded)
    data[:] = b'\xff' * len(data)
    assert canonical(decoded) == before
    with pytest.raises((AttributeError, TypeError)):
        decoded.contacts[0].collision1.name = 'altered'
    with pytest.raises((AttributeError, TypeError)):
        decoded.contacts[0].positions[0].x = 123.
    assert canonical(raw.decode_contacts_cdr(serialize_message(full_message(2)))) != before
    assert canonical(decoded) == before

def test_every_truncation_rejected_before_callback():
    data = serialize_message(full_message(1))
    for length in range(len(data)):
        with pytest.raises(raw.RawContactsDecodeError):
            raw.decode_contacts_cdr(data[:length])

@pytest.mark.parametrize('endian', ['<', '>'])
def test_all_sequence_counts_and_string_lengths_cannot_overrun(endian):
    writer = IDLEncoder(endian)
    valid = writer.write(full_message(1))
    checked = 0
    for path, kind, offset in writer.offsets:
        if path.endswith(('.count', '.strlen')):
            damaged = bytearray(valid)
            struct.pack_into(endian + 'I', damaged, offset, 0xffffffff)
            with pytest.raises(raw.RawContactsDecodeError):
                raw.decode_contacts_cdr(bytes(damaged))
            checked += 1
    assert checked >= 20

def test_string_terminator_utf8_and_trailing_bytes_fail_closed():
    writer = IDLEncoder()
    valid = writer.write(full_message(0))
    for path, kind, offset in writer.offsets:
        if kind != 'string':
            continue
        length = struct.unpack_from('<I', valid, offset - 4)[0]
        damaged = bytearray(valid)
        damaged[offset + length - 1] = 1
        with pytest.raises(raw.RawContactsDecodeError):
            raw.decode_contacts_cdr(bytes(damaged))
        if length > 1:
            damaged = bytearray(valid)
            damaged[offset] = 0xff
            with pytest.raises(raw.RawContactsDecodeError):
                raw.decode_contacts_cdr(bytes(damaged))
    with pytest.raises(raw.RawContactsDecodeError):
        raw.decode_contacts_cdr(valid + b'\x01')

@pytest.mark.parametrize('header', [b'', b'\x00', b'\x00\x01', b'\x00\x01\x00',
                                  b'\x00\x02\x00\x00', b'\xff\xff\x00\x00'])
def test_truncated_or_unsupported_encapsulation_is_explicit(header):
    with pytest.raises(raw.RawContactsDecodeError):
        raw.decode_contacts_cdr(header)

@pytest.mark.parametrize('mode', ['fine', 'stock', 'held', 'held_stock', 'empty'])
def test_each_delivery_stamps_duplicate_pairs_identity_and_reset(mode):
    pair = CallbackPair(mode)
    cases = [
        (f.NOW, (f.LEFT, f.BOOK, 3.)),
        (f.NOW, (f.BOOK, f.LEFT, 2.), (f.RIGHT, f.BOOK, 4.)),
        (f.NOW, (f.INNER, f.BOOK, 4.), (f.BOOK, f.LEFT, 5.)),
        (f.NOW + 40_000_000, (f.BOOK, f.RIGHT, 1.)),
        (f.NOW + 20_000_000, (f.LEFT, f.BOOK, 6.)),
        (f.NOW - 2_000_000_000, (f.LEFT, f.BOOK, 10.)),
        (0, (f.LEFT, f.BOOK, 2.)),
        (f.NOW + 50_000_000,),
        (f.NOW, (f.LEFT, f.OTHER_BOOK, 5.), (f.RIGHT, f.BOOK, 1.)),
        (f.NOW, (f.BOOK, f.TORSO, 8.)),
        (f.NOW, (f.BLUE, f.LEFT, 4.)),
        (f.NOW, (f.TABLE, f.INNER, 3.)),
        (f.NOW, (f.PARKED, f.BOOK, 3.), (f.LEFT, f.RIGHT, 2.)),
    ]
    for stamp, *entries in cases:
        pair.deliver(f.make_message(stamp, *entries))
    pair.reset(reset_robot_contact=True, reset_target_model=True)
    for msg in stream_messages(96):
        pair.deliver(msg)

@pytest.mark.parametrize('mode', ['stock', 'held_stock'])
@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, -0., 5e-324, 1e200])
def test_ieee_forces_reach_original_guard_unchanged(mode, value):
    pair = CallbackPair(mode)
    pair.deliver(f.make_message(f.NOW, (f.LEFT, f.BOOK, value), (f.RIGHT, f.BOOK, 1.)))

@pytest.mark.parametrize('bin_only', [False, True])
def test_first_place_hazard_empty_frames_and_attempt_replacement(bin_only):
    pair = CallbackPair('held_stock', bin_only=bin_only)
    for node in pair.nodes:
        node.table_scene_required = True
        f.activate(node, {'placement_attempt_id': 'attempt-A'})
    for msg in [f.make_message(f.NOW-1, (f.ARM, f.TABLE, 1.)),
                f.make_message(f.NOW, (f.ARM, f.TABLE, 1.), (f.LEFT, f.BOOK, 3.)),
                f.make_message(f.NOW, (f.BIN, f.ARM, 2.)), f.make_message(f.NOW)]:
        pair.deliver(msg)
    assert pair.nodes[1]._cancel.is_set()
    assert pair.nodes[1]._place_contact_guard.fault['scene_model'] == 'erc_table'
    assert len(pair.nodes[1].published) == 1
    for node, clock in zip(pair.nodes, pair.clocks):
        f.module._deactivate_place_contacts(node)
        node._cancel.clear()
        clock.nanoseconds += 1_000_000_000
        f.activate(node, {'placement_attempt_id': 'attempt-B'})
    pair.deliver(f.make_message(f.NOW, (f.ARM, f.TABLE, 1.)))
    assert pair.nodes[1]._place_contact_guard.fault is None

@pytest.mark.parametrize('release', [False, True])
def test_original_epoch_race_after_decoding_preserves_fault_precedence(release):
    pair = CallbackPair('held_stock')
    for node, ns in zip(pair.nodes, pair.namespaces):
        original = ns['_contact_force_magnitude']
        done = [False]
        def race(*args, node=node, original=original, done=done, **kwargs):
            value = original(*args, **kwargs)
            if not done[0]:
                done[0] = True
                with node._lock:
                    if release:
                        node._held_book_corners = None
                        node._transport_lock_engaged = False
                    node._clear_target_contact_samples_unlocked(reset_robot_contact=True, reset_target_model=release)
            return value
        ns['_contact_force_magnitude'] = race
    pair.deliver(f.make_message(f.NOW, (f.LEFT, f.BOOK, 2.), (f.BOOK, f.TORSO, 1.)))
    assert pair.nodes[1]._book_contact_force_samples == {}
    assert pair.nodes[1]._target_robot_contact_latched is (not release)

def test_original_callback_exception_is_not_misclassified_as_decode_error():
    node, _ = f.make_node()
    expected = RuntimeError('original callback exception')
    def failed(_):
        raise expected
    with pytest.raises(RuntimeError) as caught:
        raw.raw_contacts_callback(node, failed)(serialize_message(Contacts()))
    assert caught.value is expected
    assert not node._cancel.is_set()

@pytest.mark.parametrize('topic', ['/contacts', '/bin_contacts'])
def test_malformed_delivery_stops_without_partial_callback_and_latches_once(topic):
    node, _ = f.make_node('held_stock')
    calls = []
    receive = raw.raw_contacts_callback(node, calls.append, topic)
    receive(serialize_message(full_message(0))[:-1])
    assert not calls and node._cancel.is_set()
    assert node._held_grip_sensor_fault == 'raw_contacts_decode_failed'
    assert node._payload_hazard_latched == 'raw_contacts_decode_failed'
    first = dict(node._raw_contacts_first_failure)
    assert first['source_topic'] == topic and first['serialized_bytes'] > 4
    emitted = (len(node.events), len(node.published))
    receive(b'bad')
    assert not calls and node._raw_contacts_first_failure == first
    assert (len(node.events), len(node.published)) == emitted
    # Valid later frames are not dropped, but cannot erase the committed fault.
    receive(serialize_message(Contacts()))
    assert len(calls) == 1 and node._payload_hazard_latched == 'raw_contacts_decode_failed'

def test_decode_fault_commits_before_hold_and_status_failures():
    node, _ = f.make_node('held_stock')
    def failed_hold(_):
        assert node._lock.acquire(blocking=False)
        node._lock.release()
        assert node._cancel.is_set() and node._held_grip_sensor_fault
        raise RuntimeError('hold rejected')
    def failed_status(*args, **kwargs):
        assert node._cancel.is_set() and node._payload_hazard_latched
        raise RuntimeError('status rejected')
    node._hold_adaptive_gripper = failed_hold
    node._publish_status = failed_status
    raw.raw_contacts_callback(node, lambda _: pytest.fail('partial callback'))(b'bad')
    assert node._cancel.is_set() and node._held_grip_sensor_fault
