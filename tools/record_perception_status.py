#!/usr/bin/env python3
"""Subscribe to diagnostic strings only; record edge-triggered failure reasons.

Start before the solution publishes volatile tracking status. No robot commands
or Gazebo ground-truth subscriptions are created by this recorder.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def make_record(topic, raw, clock_ns):
    record = {'topic': topic, 'receipt_monotonic_seconds': time.monotonic(),
              'receipt_utc': datetime.now(timezone.utc).isoformat(timespec='microseconds'),
              'receipt_ros_time_ns': clock_ns}
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        record['raw'] = raw
        record['parse_error'] = 'status was not JSON'
    else:
        record['payload'] = payload
        if isinstance(payload, dict):
            record['reason'] = payload.get('reason')
            record['producer_stamp_ns'] = payload.get('stamp_ns')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--ready-file', required=True, type=Path)
    args = parser.parse_args()
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    rclpy.init()
    node = Node('erc_matrix_perception_recorder',
                parameter_overrides=[Parameter('use_sim_time', value=True)])
    with args.output.open('w', buffering=1) as stream:
        def receive(topic, message):
            record = make_record(topic, message.data, node.get_clock().now().nanoseconds)
            stream.write(json.dumps(record, sort_keys=True) + '\n')

        subscriptions = []
        for topic, durability in (
                ('/erc/perception/target_book_tracking_status', DurabilityPolicy.VOLATILE),
                ('/erc/perception/status', DurabilityPolicy.TRANSIENT_LOCAL)):
            qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                             durability=durability)
            subscriptions.append(node.create_subscription(
                String, topic, lambda message, topic=topic: receive(topic, message), qos))
        args.ready_file.write_text(json.dumps({'subscriptions_created': True,
            'monotonic_seconds': time.monotonic(), 'read_only': True}) + '\n')
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
            pass
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
