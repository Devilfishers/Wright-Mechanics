#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import NavSatFix

from mavros_msgs.srv import (
    CommandBool,
    SetMode,
    CommandTOL,
    GimbalManagerSetRoi,
)

from mavros_msgs.msg import GlobalPositionTarget

from geographiclib.geodesic import Geodesic


class DroneControl(Node):

    def __init__(self):
        super().__init__('drone_control')

        # -------------------------------------------------
        # MAVROS service clients
        # -------------------------------------------------

        self.set_mode_client = self.create_client(
            SetMode,
            '/mavros/set_mode'
        )

        self.arming_client = self.create_client(
            CommandBool,
            '/mavros/cmd/arming'
        )

        self.takeoff_client = self.create_client(
            CommandTOL,
            '/mavros/cmd/takeoff'
        )

        self.land_client = self.create_client(
            CommandTOL,
            '/mavros/cmd/land'
        )

        self.gimbal_roi_client = self.create_client(
            GimbalManagerSetRoi,
            '/mavros/gimbal_control/manager/set_roi'
        )

        # -------------------------------------------------
        # Global position setpoint publisher
        # -------------------------------------------------

        self.raw_global_pub = self.create_publisher(
            GlobalPositionTarget,
            '/mavros/setpoint_raw/global',
            10
        )

        # -------------------------------------------------
        # GPS subscriber
        # -------------------------------------------------

        self.current_gps = None
        self.home_gps = None

        self.create_subscription(
            NavSatFix,
            '/mavros/global_position/global',
            self.global_position_callback,
            qos_profile_sensor_data
        )

        self.wait_for_services()

    # =====================================================
    # GPS
    # =====================================================

    def global_position_callback(self, msg):

        self.current_gps = msg

        # Save first valid GPS fix as our home position
        if self.home_gps is None:

            self.home_gps = (
                msg.latitude,
                msg.longitude,
                msg.altitude
            )

            self.get_logger().info(
                f"Home captured: "
                f"lat={msg.latitude:.8f}, "
                f"lon={msg.longitude:.8f}, "
                f"alt={msg.altitude:.2f}"
            )

    def wait_for_gps(self, timeout=20.0):

        self.get_logger().info("Waiting for GPS...")

        start = time.time()

        while rclpy.ok():

            rclpy.spin_once(
                self,
                timeout_sec=0.1
            )

            if self.current_gps is not None:
                self.get_logger().info("GPS acquired.")
                return True

            if time.time() - start > timeout:
                self.get_logger().error(
                    "Timeout waiting for GPS."
                )
                return False

        return False

    # =====================================================
    # SERVICES
    # =====================================================

    def wait_for_services(self):

        self.get_logger().info(
            "Waiting for MAVROS services..."
        )

        self.set_mode_client.wait_for_service()
        self.arming_client.wait_for_service()
        self.takeoff_client.wait_for_service()
        self.land_client.wait_for_service()
        self.gimbal_roi_client.wait_for_service()

        self.get_logger().info(
            "MAVROS services available."
        )

    def set_mode(self, mode):

        req = SetMode.Request()
        req.custom_mode = mode

        future = self.set_mode_client.call_async(req)

        rclpy.spin_until_future_complete(
            self,
            future
        )

        result = future.result()

        if result is None:
            self.get_logger().error(
                f"Failed to set mode {mode}"
            )
            return False

        self.get_logger().info(
            f"Mode set to {mode}: "
            f"{result.mode_sent}"
        )

        return result.mode_sent

    def arm_drone(self):

        req = CommandBool.Request()
        req.value = True

        future = self.arming_client.call_async(req)

        rclpy.spin_until_future_complete(
            self,
            future
        )

        result = future.result()

        if result is None:
            self.get_logger().error(
                "Arming service failed."
            )
            return False

        self.get_logger().info(
            f"Arming result: {result.success}"
        )

        return result.success

    def takeoff(self, altitude=8.0):

        req = CommandTOL.Request()

        # This is relative takeoff altitude
        req.altitude = altitude

        future = self.takeoff_client.call_async(req)

        rclpy.spin_until_future_complete(
            self,
            future
        )

        result = future.result()

        if result is None:
            self.get_logger().error(
                "Takeoff service failed."
            )
            return False

        self.get_logger().info(
            f"Takeoff result: {result.success}"
        )

        return result.success

    def land(self):

        req = CommandTOL.Request()
        req.altitude = 0.0

        future = self.land_client.call_async(req)

        rclpy.spin_until_future_complete(
            self,
            future
        )

        result = future.result()

        if result is None:
            self.get_logger().error(
                "Landing service failed."
            )
            return False

        self.get_logger().info(
            f"Landing result: {result.success}"
        )

        return result.success

    # =====================================================
    # GIMBAL ROI
    # =====================================================

    def gimbal_roi(self, lat, lon, alt_relative_home):

        req = GimbalManagerSetRoi.Request()

        # MAVROS:
        # ROI_MODE_LOCATION = 0
        req.mode = 0

        # 0 = all / default gimbal
        req.gimbal_device_id = 0

        req.latitude = float(lat)
        req.longitude = float(lon)

        # IMPORTANT:
        # ArduPilot interprets this ROI altitude relative to home
        # for the COMMAND_LONG generated by MAVROS.
        req.altitude = float(alt_relative_home)

        future = self.gimbal_roi_client.call_async(req)

        rclpy.spin_until_future_complete(
            self,
            future
        )

        result = future.result()

        if result is None:
            self.get_logger().error(
                "ROI service call failed."
            )
            return False

        self.get_logger().info(
            f"ROI set: "
            f"lat={lat:.8f}, "
            f"lon={lon:.8f}, "
            f"alt_rel={alt_relative_home:.2f} "
            f"success={result.success}"
        )

        return result.success

    def clear_gimbal_roi(self):

        req = GimbalManagerSetRoi.Request()

        # MAVROS:
        # ROI_MODE_NONE = 3
        req.mode = 3
        req.gimbal_device_id = 0

        future = self.gimbal_roi_client.call_async(req)

        rclpy.spin_until_future_complete(
            self,
            future
        )

        result = future.result()

        if result is None:
            return False

        self.get_logger().info(
            f"ROI cleared: {result.success}"
        )

        return result.success

    # =====================================================
    # GEOGRAPHY
    # =====================================================

    def distance_to(self, lat, lon):

        if self.current_gps is None:
            return float('inf')

        inverse = Geodesic.WGS84.Inverse(
            self.current_gps.latitude,
            self.current_gps.longitude,
            lat,
            lon
        )

        # distance in meters
        return inverse['s12']

    def create_triangle(
        self,
        roi_lat,
        roi_lon,
        radius_m,
        altitude_msl
    ):

        """
        Creates three points equally spaced around ROI.

        radius_m:
            distance from ROI to each triangle corner

        altitude_msl:
            navigation altitude, NOT relative altitude
        """

        points = []

        # Bearings measured clockwise from North
        bearings = [
            0.0,
            120.0,
            240.0
        ]

        for bearing in bearings:

            result = Geodesic.WGS84.Direct(
                roi_lat,
                roi_lon,
                bearing,
                radius_m
            )

            points.append(
                (
                    result['lat2'],
                    result['lon2'],
                    altitude_msl
                )
            )

        return points

    # =====================================================
    # GLOBAL GUIDED NAVIGATION
    # =====================================================

    def create_global_target(
        self,
        lat,
        lon,
        altitude_msl
    ):

        target = GlobalPositionTarget()

        target.header.frame_id = "map"

        # IMPORTANT:
        # altitude = MSL in FRAME_GLOBAL_INT
        target.coordinate_frame = (
            GlobalPositionTarget.FRAME_GLOBAL_INT
        )

        # Position only:
        # ignore velocity, acceleration, yaw and yaw-rate.
        target.type_mask = (
            GlobalPositionTarget.IGNORE_VX |
            GlobalPositionTarget.IGNORE_VY |
            GlobalPositionTarget.IGNORE_VZ |
            GlobalPositionTarget.IGNORE_AFX |
            GlobalPositionTarget.IGNORE_AFY |
            GlobalPositionTarget.IGNORE_AFZ |
            GlobalPositionTarget.IGNORE_YAW |
            GlobalPositionTarget.IGNORE_YAW_RATE
        )

        target.latitude = float(lat)
        target.longitude = float(lon)
        target.altitude = float(altitude_msl)

        return target

    def navigate_to(
        self,
        lat,
        lon,
        altitude_msl,
        acceptance_radius=1.5,
        timeout=60.0
    ):

        target = self.create_global_target(
            lat,
            lon,
            altitude_msl
        )

        self.get_logger().info(
            f"Navigating to: "
            f"{lat:.8f}, {lon:.8f}, "
            f"alt={altitude_msl:.2f}"
        )

        start = time.time()
        last_log = 0.0

        while rclpy.ok():

            target.header.stamp = (
                self.get_clock().now().to_msg()
            )

            # Keep streaming setpoint.
            self.raw_global_pub.publish(target)

            # Allows GPS callback to update current_gps.
            rclpy.spin_once(
                self,
                timeout_sec=0.05
            )

            distance = self.distance_to(
                lat,
                lon
            )

            now = time.time()

            if now - last_log > 1.0:

                self.get_logger().info(
                    f"Distance to target: "
                    f"{distance:.2f} m"
                )

                last_log = now

            if distance <= acceptance_radius:

                self.get_logger().info(
                    f"Waypoint reached "
                    f"({distance:.2f} m)"
                )

                return True

            if now - start > timeout:

                self.get_logger().error(
                    f"Waypoint timeout. "
                    f"Remaining distance: "
                    f"{distance:.2f} m"
                )

                return False

            time.sleep(0.05)

        return False

    # =====================================================
    # TRIANGLE MISSION
    # =====================================================

    def fly_triangle_around_roi(
        self,
        roi_lat,
        roi_lon,
        roi_alt_relative,
        radius_m,
        altitude_msl,
        close_triangle=True
    ):

        triangle = self.create_triangle(
            roi_lat,
            roi_lon,
            radius_m,
            altitude_msl
        )

        self.get_logger().info(
            "Triangle generated:"
        )

        for i, point in enumerate(triangle):

            self.get_logger().info(
                f"Corner {i + 1}: "
                f"{point[0]:.8f}, "
                f"{point[1]:.8f}"
            )

        # Lock gimbal before flying triangle
        if not self.gimbal_roi(
            roi_lat,
            roi_lon,
            roi_alt_relative
        ):
            self.get_logger().error(
                "Could not set ROI."
            )
            return False

        # Visit three unique triangle corners
        for i, point in enumerate(triangle):

            self.get_logger().info(
                f"Flying to triangle corner "
                f"{i + 1}/3"
            )

            success = self.navigate_to(
                point[0],
                point[1],
                point[2]
            )

            if not success:
                return False

            # Stay briefly at each corner.
            time.sleep(2.0)

        # Return to first corner to actually close triangle.
        if close_triangle:

            self.get_logger().info(
                "Closing triangle..."
            )

            first = triangle[0]

            if not self.navigate_to(
                first[0],
                first[1],
                first[2]
            ):
                return False

        return True


def main(args=None):

    rclpy.init(args=args)

    drone = DroneControl()

    # =====================================================
    # CONFIGURATION
    # =====================================================

    # Replace these with the GPS position corresponding
    # to the person/car in Gazebo.
    ROI_LAT = -35.36326036 
    ROI_LON = 149.16534361

    # ROI altitude is relative to HOME.
    #
    # For a standing person or car on the ground:
    # ~0.8 - 1.2 m aims approximately at the body / vehicle.
    ROI_ALT_REL_HOME = 1.0

    # Radius from ROI to each triangle corner.
    TRIANGLE_RADIUS_M = 7

    # Takeoff / cruising height above home.
    FLIGHT_HEIGHT_REL = 10.0
    
    
    flight_alt_msl = 590.0
    # =====================================================
    # GET HOME
    # =====================================================

    if not drone.wait_for_gps():
        drone.destroy_node()
        rclpy.shutdown()
        return

    home_lat = drone.home_gps[0]
    home_lon = drone.home_gps[1]
    home_alt_msl = drone.home_gps[2]


    

    drone.get_logger().info(
        f"Home MSL altitude: "
        f"{home_alt_msl:.2f} m"
    )

    drone.get_logger().info(
        f"Flight MSL altitude: "
        f"{flight_alt_msl:.2f} m"
    )

    # =====================================================
    # START FLIGHT
    # =====================================================

    if not drone.set_mode('GUIDED'):
        return

    time.sleep(1)

    if not drone.arm_drone():
        return

    time.sleep(1)

    if not drone.takeoff(
        FLIGHT_HEIGHT_REL
    ):
        return

    # Allow takeoff to settle.
    time.sleep(8)

    # =====================================================
    # TRIANGLE + ROI
    # =====================================================

    mission_ok = drone.fly_triangle_around_roi(
        roi_lat=ROI_LAT,
        roi_lon=ROI_LON,
        roi_alt_relative=ROI_ALT_REL_HOME,
        radius_m=TRIANGLE_RADIUS_M,
        altitude_msl=flight_alt_msl,
        close_triangle=True
    )

    if not mission_ok:

        drone.get_logger().error(
            "Triangle mission failed. "
            "Switching to RTL."
        )

        drone.set_mode('RTL')

        drone.destroy_node()
        rclpy.shutdown()
        return

    # =====================================================
    # FINISHED OBSERVATION
    # =====================================================

    drone.clear_gimbal_roi()

    time.sleep(1)

    # =====================================================
    # RETURN TO HOME AT CRUISE ALTITUDE
    # =====================================================

    drone.get_logger().info(
        "Returning to home position..."
    )

    home_reached = drone.navigate_to(
        home_lat,
        home_lon,
        flight_alt_msl,
        acceptance_radius=1.5,
        timeout=90.0
    )

    if home_reached:

        drone.get_logger().info(
            "Home reached. Landing..."
        )

        drone.land()

    else:

        # Backup if our setpoint return fails
        drone.get_logger().warning(
            "Could not reach home with setpoint. "
            "Using RTL as fallback."
        )

        drone.set_mode('RTL')

    # Keep node alive for a little while so
    # LAND / RTL command can be processed.
    end_time = time.time() + 5.0

    while rclpy.ok() and time.time() < end_time:
        rclpy.spin_once(
            drone,
            timeout_sec=0.1
        )

    drone.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
