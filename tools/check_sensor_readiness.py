#!/usr/bin/env python3
"""Read-only ROS sensor readiness probe; publish nothing and exit within 60 s."""

import argparse
from collections import deque
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, JointState, LaserScan


TOPICS = {
    '/clock': Clock,
    '/odom': Odometry,
    '/joint_states': JointState,
    '/scan_front_raw': LaserScan,
    '/scan_rear_raw': LaserScan,
    '/head_front_camera/head_front_camera/color/image_raw': Image,
    '/head_front_camera/head_front_camera/depth/image_rect_raw': Image,
    '/head_front_camera/head_front_camera/depth/camera_info': CameraInfo,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--recent-seconds', type=float, default=5.0)
    args = parser.parse_args()
    timeout = min(60.0, max(1.0, args.timeout))
    recent_seconds = max(0.1, args.recent_seconds)
    rclpy.init()
    node = Node('erc_demo_readiness_probe')
    samples = {topic: deque(maxlen=2) for topic in TOPICS}
    counts = {topic: 0 for topic in TOPICS}
    qos = QoSProfile(
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )

    def receive(topic, message):
        stamp = message.clock if topic == '/clock' else message.header.stamp
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        counts[topic] += 1
        samples[topic].append((time.monotonic(), stamp_ns))

    subscriptions = [
        node.create_subscription(
            msg_type, topic,
            lambda message, topic=topic: receive(topic, message), qos,
        )
        for topic, msg_type in TOPICS.items()
    ]

    def problems(now):
        result = {}
        for topic, data in samples.items():
            if len(data) < 2:
                result[topic] = f'only {len(data)} samples'
            elif now - data[0][0] > recent_seconds:
                result[topic] = 'no two recent samples'
            elif data[1][1] <= data[0][1]:
                result[topic] = 'timestamps not advancing'
        return result

    started = time.monotonic()
    next_progress = started
    passed = False
    missing = dict.fromkeys(TOPICS, 'no samples')
    print(f'Checking {len(TOPICS)} sensor topics for up to {timeout:.0f} s...', flush=True)
    try:
        while rclpy.ok() and time.monotonic() - started < timeout:
            rclpy.spin_once(node, timeout_sec=0.2)
            now = time.monotonic()
            missing = problems(now)
            if not missing:
                passed = True
                break
            if now >= next_progress:
                print(
                    f'Waiting ({now - started:.1f} s): '
                    + '; '.join(f'{topic}: {reason}' for topic, reason in missing.items()),
                    flush=True,
                )
                next_progress = now + 5.0
        print(json.dumps({
            'passed': passed,
            'wall_seconds': round(time.monotonic() - started, 3),
            'read_only': True,
            'counts': counts,
            'missing': missing,
        }, sort_keys=True), flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
