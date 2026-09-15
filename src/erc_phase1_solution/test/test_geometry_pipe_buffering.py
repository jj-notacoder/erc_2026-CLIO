"""Framing and bounded reads through the production worker reader."""
import io
import queue
from types import SimpleNamespace

import pytest

from erc_phase1_solution.geometry_process_pool import _Worker
from erc_phase1_solution.geometry_process_protocol import MAX_FRAME_BYTES, encode_frame


class CountingRaw(io.RawIOBase):
    def __init__(self, content, chunk=65536):
        self.content = content
        self.offset = 0
        self.calls = 0
        self.chunk = chunk

    def readable(self):
        return True

    def readinto(self, target):
        self.calls += 1
        count = min(len(target), self.chunk, len(self.content)-self.offset)
        target[:count] = self.content[self.offset:self.offset+count]
        self.offset += count
        return count


def read(content, chunk=65536):
    raw = CountingRaw(content, chunk)
    worker = _Worker.__new__(_Worker)
    worker.number = 0
    worker.events = queue.Queue(maxsize=32)
    worker.fault = None
    worker.process = SimpleNamespace(stdout=raw)
    worker.threads = []
    worker.start()
    worker._read()
    events = []
    while not worker.events.empty():
        events.append(worker.events.get_nowait())
    worker.process.stdout.close()
    assert raw.closed
    return worker, raw.calls, events


@pytest.mark.parametrize('chunk', [1, 7, 65536])
def test_complete_fragmented_and_duplicate_frames_keep_exact_order(chunk):
    values = [dict(kind='test', payload='x'*2000), dict(kind='test', payload='next')]
    content = b''.join(encode_frame(v) for v in [values[0], values[0], values[1]])
    worker, calls, events = read(content,chunk)
    assert events == [(0,values[0]),(0,values[0]),(0,values[1]),(0,None)]
    assert worker.fault == 'geometry output pipe closed'
    if chunk == 65536:
        assert calls <= 2  # A whole frame no longer takes a raw read per byte.


@pytest.mark.parametrize('content', [b'{bad}\n', b'{"x":1}', b'x'*(MAX_FRAME_BYTES+2)])
def test_bad_truncated_or_overlong_frames_never_emit_a_result(content):
    worker, calls, events = read(content)
    assert worker.fault is not None and events == [(0,None)]
    assert calls <= MAX_FRAME_BYTES//65536+3


def test_complete_frame_before_truncated_frame_preserves_only_complete_prefix():
    value = {'kind':'prefix'}
    worker, _, events = read(encode_frame(value)+b'{"bad":')
    assert events == [(0,value),(0,None)]
    assert worker.fault == 'truncated geometry output'
