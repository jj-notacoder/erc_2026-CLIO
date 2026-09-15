#!/usr/bin/env python3
"""Save one live ROS image for simulator diagnostics and report preparation."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument(
        '--topic',
        default='/head_front_camera/head_front_camera/color/image_raw',
    )
    parser.add_argument('--timeout', type=float, default=15.0)
    arguments = parser.parse_args()

    rclpy.init()
    node = Node('erc_capture_camera_once')
    bridge = CvBridge()
    received = []
    qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )

    def receive(message: Image) -> None:
        if received:
            return
        image = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(arguments.output), image):
            raise OSError(f'could not write {arguments.output}')
        received.append(message.header.stamp)

    subscription = node.create_subscription(Image, arguments.topic, receive, qos)
    deadline = time.monotonic() + max(0.1, arguments.timeout)
    try:
        while not received and time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()

    if not received:
        print(f'No image received from {arguments.topic}', flush=True)
        return 2
    stamp = received[0]
    print(
        f'Saved {arguments.output} from ROS stamp {stamp.sec}.{stamp.nanosec:09d}',
        flush=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
