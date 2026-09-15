#!/usr/bin/env python3
"""Temporary publisher for same-world manipulation probes; remove before commit."""

import json
import time

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


def main():
    rclpy.init()
    node = Node('erc_probe_publisher')
    point_qos = QoSProfile(depth=1)
    point_qos.reliability = ReliabilityPolicy.RELIABLE
    point_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    point_pub = node.create_publisher(
        PointStamped,
        '/erc/perception/target_book',
        point_qos,
    )
    command_pub = node.create_publisher(
        String,
        '/erc/manipulation/command',
        QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
    )
    deadline = time.monotonic() + 5.0
    while (
        point_pub.get_subscription_count() < 1
        or command_pub.get_subscription_count() < 1
    ) and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    point = PointStamped()
    point.header.frame_id = 'base_footprint'
    point.point.x = 0.6937794636
    point.point.y = -0.0549325514
    point.point.z = 1.5841858155
    print(
        f'subscribers point={point_pub.get_subscription_count()} '
        f'command={command_pub.get_subscription_count()}',
        flush=True,
    )
    for _ in range(12):
        point.header.stamp = node.get_clock().now().to_msg()
        point_pub.publish(point)
        rclpy.spin_once(node, timeout_sec=0.05)
    command = String()
    command.data = json.dumps({'event': 'pick'})
    for _ in range(8):
        command_pub.publish(command)
        rclpy.spin_once(node, timeout_sec=0.10)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
