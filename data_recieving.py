#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class DetectedObjectsSubscriber(Node):

    def __init__(self):
        super().__init__('detected_objects_subscriber')

        self.subscription = self.create_subscription(
            String,
            '/detected_objects',
            self.detection_callback,
            10
        )

        self.get_logger().info(
            'Listening to /detected_objects...'
        )

    def detection_callback(self, msg):

        self.get_logger().info(
            f'Detected object message: {msg.data}'
        )


def main(args=None):

    rclpy.init(args=args)

    node = DetectedObjectsSubscriber()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
