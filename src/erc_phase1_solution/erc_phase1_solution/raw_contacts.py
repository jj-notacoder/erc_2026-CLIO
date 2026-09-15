"""Exact Contacts CDR1 views; stateful safety decisions stay in the owner.

Supported schema is the pinned Humble ros_gz_interfaces Contacts/Contact/
JointWrench/Entity IDL. Decode the entire message before exposing a callback.
No force arithmetic, contact filtering, history or epoch logic belongs here.
"""
from collections import namedtuple
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import operator
import struct


class RawContactsDecodeError(ValueError):
    """Unsupported representation or malformed complete Contacts message."""


_Stamp = namedtuple('_Stamp', 'sec nanosec')
_Header = namedtuple('_Header', 'stamp frame_id')
_Entity = namedtuple('_Entity', 'id name type')
_String = namedtuple('_String', 'data')
_UInt32 = namedtuple('_UInt32', 'data')
_Contact = namedtuple('_Contact', 'collision1 collision2 positions normals depths wrenches')
_Contacts = namedtuple('_Contacts', 'header contacts')


@dataclass(frozen=True, slots=True)
class _Vector3:
    _data: bytes
    _offset: int
    _double: struct.Struct

    @property
    def x(self):
        return self._double.unpack_from(self._data, self._offset)[0]

    @property
    def y(self):
        return self._double.unpack_from(self._data, self._offset + 8)[0]

    @property
    def z(self):
        return self._double.unpack_from(self._data, self._offset + 16)[0]


@dataclass(frozen=True, slots=True)
class _Wrench:
    _data: bytes
    _offset: int
    _double: struct.Struct

    @property
    def force(self):
        return _Vector3(self._data, self._offset, self._double)

    @property
    def torque(self):
        return _Vector3(self._data, self._offset + 24, self._double)


@dataclass(frozen=True, slots=True)
class _JointWrench:
    header: _Header
    _body_1_name: str
    _body_1_id: int
    _body_2_name: str
    _body_2_id: int
    _data: bytes
    _offset: int
    _double: struct.Struct

    @property
    def body_1_name(self):
        return _String(self._body_1_name)

    @property
    def body_1_id(self):
        return _UInt32(self._body_1_id)

    @property
    def body_2_name(self):
        return _String(self._body_2_name)

    @property
    def body_2_id(self):
        return _UInt32(self._body_2_id)

    @property
    def body_1_wrench(self):
        return _Wrench(self._data, self._offset, self._double)

    @property
    def body_2_wrench(self):
        return _Wrench(self._data, self._offset + 48, self._double)


@dataclass(frozen=True, slots=True)
class _NumericSequence(Sequence):
    _data: bytes
    _offset: int
    _count: int
    _double: struct.Struct
    _vectors: bool

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(self._count)))
        index = operator.index(index)
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        offset = self._offset + index * (24 if self._vectors else 8)
        if self._vectors:
            return _Vector3(self._data, offset, self._double)
        return self._double.unpack_from(self._data, offset)[0]


class _Cursor:
    __slots__ = ('data', 'offset', 'end', 'i32', 'u32', 'u64', 'double')

    def __init__(self, data, endian):
        self.data, self.offset, self.end = data, 4, len(data)
        self.i32 = struct.Struct(endian + 'i')
        self.u32 = struct.Struct(endian + 'I')
        self.u64 = struct.Struct(endian + 'Q')
        self.double = struct.Struct(endian + 'd')

    def take(self, size, alignment=1):
        # CDR1 alignment origin follows the four-byte encapsulation header.
        start = 4 + ((self.offset - 4 + alignment - 1) // alignment) * alignment
        if size < 0 or start > self.end or size > self.end - start:
            raise RawContactsDecodeError(f'truncated CDR field at offset {self.offset}')
        self.offset = start + size
        return start

    def number(self, reader):
        return reader.unpack_from(self.data, self.take(reader.size, reader.size))[0]

    def string(self):
        count = self.number(self.u32)
        if count < 1:
            raise RawContactsDecodeError('CDR string has no terminator')
        start = self.take(count)
        if self.data[start + count - 1] != 0:
            raise RawContactsDecodeError('CDR string terminator is missing')
        try:
            return self.data[start:start + count - 1].decode('utf-8', errors='strict')
        except UnicodeDecodeError as error:
            raise RawContactsDecodeError(f'invalid UTF-8 at offset {start}') from error

    def header(self):
        return _Header(_Stamp(self.number(self.i32), self.number(self.u32)), self.string())

    def entity(self):
        identity, name = self.number(self.u64), self.string()
        kind = self.data[self.take(1)]
        return _Entity(identity, name, kind)

    def count(self):
        count = self.number(self.u32)
        # Every variable element in this fixed schema occupies at least one
        # byte. Do not allocate from an untrusted count before byte bounds.
        if count > self.end - self.offset:
            raise RawContactsDecodeError('sequence count exceeds remaining bytes')
        return count

    def numeric_sequence(self, vectors):
        count = self.number(self.u32)
        width = 24 if vectors else 8
        # Empty sequences do not advance alignment for absent elements.
        offset = self.take(count * width, 8) if count else self.offset
        return _NumericSequence(self.data, offset, count, self.double, vectors)

    def joint_wrench(self):
        header = self.header()
        name1, identity1 = self.string(), self.number(self.u32)
        name2, identity2 = self.string(), self.number(self.u32)
        offset = self.take(96, 8)
        return _JointWrench(header, name1, identity1, name2, identity2,
                            self.data, offset, self.double)

    def contact(self):
        first, second = self.entity(), self.entity()
        positions = self.numeric_sequence(True)
        normals = self.numeric_sequence(True)
        depths = self.numeric_sequence(False)
        wrenches = tuple(self.joint_wrench() for _ in range(self.count()))
        return _Contact(first, second, positions, normals, depths, wrenches)


def decode_contacts_cdr(data):
    """Return owned immutable full-schema values without ROS object creation.

    Preserve all IEEE binary64 values, names, IDs, sequence order and both
    wrench bodies. Nonfinite scalar values are data, not a codec error: the
    unchanged safety callback owns their existing fault handling.
    """
    if type(data) is bytes:
        owned = data
    elif isinstance(data, (bytearray, memoryview)):
        owned = bytes(data)
    else:
        raise RawContactsDecodeError('raw Contacts input must be bytes-like')
    if len(owned) < 4:
        raise RawContactsDecodeError('missing CDR encapsulation')
    representation = int.from_bytes(owned[:2], 'big')
    if representation not in (0, 1) or owned[2:4] != b'\x00\x00':
        raise RawContactsDecodeError('unsupported CDR representation/options')
    reader = _Cursor(owned, '<' if representation == 1 else '>')
    header = reader.header()
    contacts = tuple(reader.contact() for _ in range(reader.count()))
    if reader.offset != reader.end:
        raise RawContactsDecodeError('unexpected trailing CDR bytes')
    return _Contacts(header, contacts)


def _stop_on_decode_failure(node, data, error, source_topic):
    """A wire failure is a sensor fault, never a fabricated contact verdict."""
    reason = 'raw_contacts_decode_failed'
    # Cancel even if acquiring the normal command guard or reporting fails.
    node._cancel.set()
    raw = data if type(data) is bytes else b''
    diagnostic = dict(reason=reason, source_topic=str(source_topic),
                      error_type=type(error).__name__, error=str(error)[:256],
                      serialized_bytes=len(raw), raw_sha256=hashlib.sha256(raw).hexdigest(),
                      raw_prefix_hex=raw[:256].hex(),
                      raw_prefix_truncated=len(raw) > 256)
    with node._adaptive_command_guard():
        with node._lock:
            # A concurrent new-command admission may have cleared the early
            # event before this lock. Commit it again with the permanent latch.
            node._cancel.set()
            node._held_grip_sensor_fault = (getattr(node, '_held_grip_sensor_fault', None) or reason)
            if getattr(node, '_adaptive_close_active', False):
                node._adaptive_overload_latched = (getattr(node, '_adaptive_overload_latched', None) or reason)
            if (getattr(node, '_held_book_corners', None) is not None
                    or getattr(node, '_transport_lock_engaged', False)):
                node._payload_hazard_latched = (getattr(node, '_payload_hazard_latched', None) or reason)
            first = getattr(node, '_raw_contacts_first_failure', None) is None
            if first:
                node._raw_contacts_first_failure = dict(diagnostic)
        # No sensor lock is held by the existing measured hold helper.
        if first:
            try:
                node._adaptive_hold_sent = False
                node._hold_adaptive_gripper(reason)
            except Exception as hold_error:
                diagnostic['hold_error'] = type(hold_error).__name__
            try:
                node._publish_status('payload_hazard', **diagnostic)
            except Exception:
                pass  # Cancellation and persistent faults are already committed.


def raw_contacts_callback(node, callback, source_topic='/contacts'):
    """Decode synchronously before the original callback and all its locks.

    Unsupported formats deliberately stop. No permissive stock-deserializer
    fallback can silently accept a malformed message or fabricate an empty one.
    """
    def receive(data):
        try:
            message = decode_contacts_cdr(data)
        except Exception as error:
            _stop_on_decode_failure(node, data, error, source_topic)
            return None
        return callback(message)
    return receive
