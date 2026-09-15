"""Optional completed-frame evidence for leaving an unproductive marker view.

No frame, target or clock error is evidence of absence. This contract only
accelerates the existing bounded search turn; it never certifies shelf geometry.
"""
from dataclasses import dataclass
from collections import deque
import math
import re

MAX_AGE_NS = 1_000_000_000
MAX_SKEW_NS = 200_000_000
MAX_STAMP_NS = (1 << 63) - 1
OUTCOMES = frozenset(('target_absent', 'target_present_pending', 'depth_rejected',
                      'cloud_published', 'processing_error', 'frame_invalid'))


def _integer(value, low=1, high=MAX_STAMP_NS):
    return type(value) is int and low <= value <= high


@dataclass(frozen=True)
class SearchRequest:
    trial_id: str
    request_id: int
    target_column: int
    target_colour: str
    not_before_ns: int

    def fields(self):
        return dict(trial_id=self.trial_id, marker_search_request_id=self.request_id,
                    shelf_column_number=self.target_column, book_colour=self.target_colour,
                    marker_search_not_before_ns=self.not_before_ns)


def request_from_fields(fields):
    """Malformed optional metadata disables early turning, never ordinary search."""
    if not isinstance(fields, dict):
        return None
    trial = fields.get('trial_id')
    request = fields.get('marker_search_request_id')
    column, colour = fields.get('shelf_column_number'), fields.get('book_colour')
    stamp = fields.get('marker_search_not_before_ns')
    if (not isinstance(trial, str) or re.fullmatch(r'[0-9a-f]{12}', trial) is None
            or not _integer(request, high=(1 << 31)-1) or not _integer(column, high=5)
            or colour not in ('red', 'blue', 'green', 'yellow') or not _integer(stamp)):
        return None
    return SearchRequest(trial, request, column, colour, stamp)


def receive_request(node, payload):
    """Bind only the already accepted marker mode and target; retain no old epoch."""
    previous = getattr(node, '_marker_search_request', None)
    if type(previous) is not SearchRequest:
        previous = None
    request = request_from_fields(payload)
    if (request is None or node.mode != 'markers'
            or (request.target_column, request.target_colour) != (node.target_column, node.target_colour)):
        request = None
    elif (previous is not None and previous.trial_id == request.trial_id
          and request.request_id <= previous.request_id and request != previous):
        request = None
    if request != previous:
        node._marker_search_frame_sequence = 0
    node._marker_search_request = request


def capture_frame(node):
    """Capture the exact inputs of this serial perception callback before detection."""
    request = getattr(node, '_marker_search_request', None)
    if type(request) is not SearchRequest:
        return None
    node._marker_search_frame_sequence = getattr(node, '_marker_search_frame_sequence', 0)+1
    sequence = node._marker_search_frame_sequence
    try:
        rgb = node.latest_rgb_message.header.stamp
        depth = node.latest_depth_message.header.stamp
        rgb_ns = int(rgb.sec)*1_000_000_000+int(rgb.nanosec)
        depth_ns = int(depth.sec)*1_000_000_000+int(depth.nanosec)
        age = float(node.maximum_frame_age)
        skew = float(node.maximum_frame_skew)
        eligible = bool(node._ready() and math.isfinite(age) and 0 < age <= 60
                        and math.isfinite(skew) and 0 <= skew <= 60)
        limit = min(MAX_AGE_NS, int(age*1e9)) if eligible else 0
        return request, sequence, rgb_ns, depth_ns, limit, min(MAX_SKEW_NS, int(skew*1e9)) if eligible else 0, eligible
    except Exception:
        return request, sequence, 0, 0, 0, 0, False


def complete_frame(node, frame, outcome, target_count):
    """Publish only after this frame's detector/outcome path completed.

    No publication failure may affect the original positive detection or its
    exception. An invalid frame can reset a streak but can never supply absence.
    """
    if frame is None:
        return
    try:
        request, sequence, rgb_ns, depth_ns, limit, skew, eligible = frame
        if (getattr(node, '_marker_search_request', None) != request or node.mode != 'markers'
                or (node.target_column, node.target_colour) != (request.target_column, request.target_colour)):
            return
        now = int(node.get_clock().now().nanoseconds)
        valid_limit = _integer(limit, high=MAX_AGE_NS)
        expires = (min(rgb_ns, depth_ns)+limit
                   if valid_limit and _integer(rgb_ns) and _integer(depth_ns) else 0)
        fresh = (eligible and valid_limit and _integer(expires)
                 and _integer(now) and _integer(rgb_ns) and _integer(depth_ns)
                 and min(rgb_ns, depth_ns) > request.not_before_ns
                 and 0 <= now-rgb_ns <= limit and 0 <= now-depth_ns <= limit
                 and abs(rgb_ns-depth_ns) <= skew)
        valid_count = _integer(target_count, low=0)
        frame_valid = bool(fresh and outcome in OUTCOMES and valid_count
                           and outcome not in ('processing_error', 'frame_invalid'))
        node._publish_status('marker_search_frame_completed', **request.fields(),
            schema=1, frame_sequence=sequence, rgb_stamp_ns=rgb_ns, depth_stamp_ns=depth_ns,
            completed_ros_ns=now, frame_valid=frame_valid, outcome=outcome,
            frame_age_limit_ns=limit, frame_valid_until_ns=expires,
            target_candidate_count=target_count)
    except Exception:
        pass


class NegativeFrameGate:
    """Constant storage; three consecutive distinct fresh negative completions."""
    def __init__(self, request):
        if type(request) is not SearchRequest:
            raise ValueError('marker search request is invalid')
        self.request = request
        self.count = 0
        self.last_sequence = 0
        self.last_rgb_ns = self.last_depth_ns = self.last_completed_ns = 0
        self.frames = deque(maxlen=3)

    def reset(self):
        self.count = 0
        self.frames.clear()

    def _frames_fresh(self, now_ns):
        # Carry each captured configured age limit through transport and ticks.
        return bool(_integer(now_ns)
                    and all(completed <= now_ns <= expires
                            and 0 <= now_ns-rgb <= MAX_AGE_NS
                            and 0 <= now_ns-depth <= MAX_AGE_NS
                            for rgb, depth, completed, expires in self.frames))

    def observe(self, event, now_ns):
        if not isinstance(event, dict):
            self.reset()
            return
        request = request_from_fields(event)
        sequence = event.get('frame_sequence')
        rgb, depth, completed = (event.get(k) for k in
                                 ('rgb_stamp_ns', 'depth_stamp_ns', 'completed_ros_ns'))
        age_limit, expires = (event.get(k) for k in
                              ('frame_age_limit_ns', 'frame_valid_until_ns'))
        valid = (event.get('event') == 'marker_search_frame_completed'
            and event.get('mode') == 'markers' and type(event.get('schema')) is int and event['schema'] == 1
            and request == self.request and _integer(sequence) and _integer(now_ns)
            and all(_integer(s) for s in (rgb, depth, completed))
            and _integer(age_limit, high=MAX_AGE_NS) and _integer(expires)
            and expires == min(rgb, depth)+age_limit and now_ns <= expires
            and min(rgb, depth) > self.request.not_before_ns
            and 0 <= now_ns-rgb <= MAX_AGE_NS and 0 <= now_ns-depth <= MAX_AGE_NS
            and abs(rgb-depth) <= MAX_SKEW_NS
            and max(rgb, depth) <= completed <= now_ns
            and event.get('frame_valid') is True
            and isinstance(event.get('outcome'), str) and event['outcome'] in OUTCOMES
            and _integer(event.get('target_candidate_count'), low=0))
        if not valid:
            self.reset()
            return
        if (sequence <= self.last_sequence or rgb <= self.last_rgb_ns
                or depth <= self.last_depth_ns or completed < self.last_completed_ns):
            self.reset()
            return
        if sequence != self.last_sequence+1:
            self.reset()
        self.last_sequence, self.last_rgb_ns, self.last_depth_ns, self.last_completed_ns = sequence, rgb, depth, completed
        if event['outcome'] == 'target_absent' and event['target_candidate_count'] == 0:
            if not self._frames_fresh(now_ns):
                self.reset()
            self.frames.append((rgb, depth, completed, expires))
            self.count = len(self.frames)
        else:
            self.reset()

    def ready(self, now_ns):
        return bool(self.count == 3 and self._frames_fresh(now_ns))
