#!/usr/bin/env python3

import rospy
import math
import time
from geometry_msgs.msg import PoseStamped, Point, Twist
from visualization_msgs.msg import Marker, MarkerArray
import tf


class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = time.time()

    def calculate(self, error):
        current_time = time.time()
        delta_time = current_time - self.previous_time
        delta_error = error - self.previous_error

        # Calculate integral and derivative terms
        self.integral += error * delta_time
        derivative = delta_error / delta_time if delta_time > 0 else 0.0

        # Compute PID output
        output = (self.Kp * error) + (self.Ki * self.integral) + \
            (self.Kd * derivative)

        # Update previous values
        self.previous_error = error
        self.previous_time = current_time

        return output


class DronePathFollower:
    def __init__(self):
        rospy.init_node('drone_path_follower', anonymous=True)

        # PID parameters for smooth movement on each axis
        self.pid_x = PIDController(Kp=0.8, Ki=0.0, Kd=0.3)
        self.pid_y = PIDController(Kp=0.8, Ki=0.0, Kd=0.3)
        self.pid_z = PIDController(Kp=0.4, Ki=0.0, Kd=0.3)

        # Initialize variables
        self.current_index = 0
        self.goal = None
        self.new_goal_pending = False
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.path_marker_pub = rospy.Publisher(
            '/path_marker', MarkerArray, queue_size=10)

        # Set up scaling factor
        self.scaling_factor = 0.05  # Default scaling factor; modify as needed

        # Define the path (scaled)
        self.global_path = self.generate_scaled_path()

        # Subscribe to topics for current position and goal
        rospy.Subscriber('/ground_truth_to_tf/pose',
                         PoseStamped, self.current_pose_callback)
        rospy.Subscriber('/move_base_simple/goal',
                         PoseStamped, self.goal_callback)

        # Initialize other required variables
        self.listener = tf.TransformListener()
        self.current_pose = None
        self.goal_threshold = 0.05
        self.waypoint_threshold = 0.1
        self.max_velocity = 0.6

    def generate_scaled_path(self):
        # raw_path = [[89, 93, 56], [90, 92, 56], [91, 91, 56], [92, 90, 56], [93, 89, 56], [94, 88, 56], [95, 87, 56], [96, 86, 56], [97, 85, 56], [98, 84, 56], [99, 83, 56], [100, 82, 56], [101, 81, 56], [102, 80, 55], [103, 79, 55], [104, 78, 55], [105, 77, 54], [106, 76, 53], [
        #     107, 75, 53], [108, 74, 52], [109, 73, 52], [110, 72, 52], [111, 71, 51], [112, 70, 51], [113, 69, 50], [114, 69, 50], [115, 68, 49], [116, 68, 49], [117, 67, 49], [118, 67, 49], [119, 66, 48], [120, 66, 48], [121, 65, 47], [122, 65, 47], [123, 65, 47], [124, 64, 46],
        # [124, 64, 46], [125, 65, 46], [126, 66, 46], [127, 67, 46], [128, 67, 46], [129, 67, 46], [130, 68, 46], [131, 68, 46], [132, 69, 46], [133, 70, 46], [134, 70, 46], [135, 70, 46], [136, 71, 46], [137, 71, 46], [138, 72, 46], [139, 72, 46], [140, 73, 46], [141, 73, 46], [142, 73, 46], [
        #     143, 74, 46], [144, 75, 46], [145, 75, 46], [146, 75, 46], [147, 76, 46], [148, 76, 46], [149, 77, 46], [150, 77, 46], [151, 78, 46], [152, 78, 46], [153, 79, 46], [154, 79, 46], [155, 79, 46], [156, 80, 46], [157, 80, 46], [158, 81, 46], [159, 81, 46], [160, 82, 46], [161, 82, 46],
        # [162, 83, 46], [163, 83, 46], [164, 84, 46], [165, 84, 46], [166, 84, 46], [167, 85, 46], [168, 85, 46], [169, 86, 46], [170, 86, 46], [171, 87, 46], [172, 87, 46], [173, 88, 46], [174, 88, 46], [175, 89, 46], [176, 89, 46], [177, 90, 46], [178, 90, 46], [179, 90, 46], [180, 91, 46], [
        #     181, 91, 46], [182, 92, 46], [183, 92, 46], [184, 93, 46], [185, 93, 46], [186, 93, 46], [187, 94, 46], [188, 94, 46], [189, 95, 46], [190, 95, 46], [191, 96, 46], [192, 96, 46], [193, 97, 46], [194, 97, 46], [195, 98, 46], [196, 98, 46], [197, 99, 45], [198, 99, 45], [199, 99, 45], [200, 100, 44],
        # [200, 100, 44], [200, 101, 44], [200, 102, 44], [200, 103, 44], [200, 104, 44], [200, 105, 44], [200, 106, 44], [200, 107, 44], [200, 108, 44], [200, 109, 44], [200, 110, 44], [200, 111, 44], [200, 112, 44], [200, 113, 44], [200, 114, 44], [200, 115, 44], [200, 116, 44], [200, 117, 44], [200, 118, 44], [200, 119, 44], [200, 120, 44], [200, 121, 44], [200, 122, 44], [200, 123, 44], [200, 124, 44], [200, 125, 44], [200, 126, 44], [200, 127, 44], [200, 128, 44], [200, 129, 44], [200, 130, 44], [200, 131, 44], [200, 132, 44], [200, 133, 44], [200, 134, 44], [200, 135, 44], [200, 136, 44], [200, 137, 44], [200, 138, 44], [200, 139, 44], [
        #     200, 140, 44], [200, 141, 44], [200, 142, 44], [200, 143, 44], [200, 144, 44], [200, 145, 44], [200, 146, 44], [200, 147, 44], [200, 148, 44], [200, 149, 44], [200, 150, 44], [200, 151, 44], [200, 152, 44], [200, 153, 44], [200, 154, 44], [200, 155, 44], [200, 156, 44], [200, 157, 44], [200, 158, 44], [200, 159, 44], [200, 160, 44], [200, 161, 44], [200, 162, 44], [200, 163, 44], [200, 164, 44], [200, 165, 45], [200, 166, 45], [199, 167, 46], [199, 168, 46], [198, 169, 47], [198, 170, 47], [197, 171, 48], [197, 172, 48], [197, 173, 49], [197, 174, 49], [196, 175, 50], [196, 176, 50], [195, 177, 51], [195, 178, 51], [195, 179, 51], [194, 180, 52]]

        #Planner modificao
        raw_path = [
            [88, 94, 56], [89, 93, 56], [90, 92, 56], [91, 91, 56], [92, 90, 56], [93, 89, 56], [94, 88, 56], [95, 87, 56], [96, 86, 56], [97, 85, 56], [98, 84, 56], [99, 83, 56], [100, 82, 56], [101, 81, 56], [102, 80, 55], [103, 79, 55], [104, 78, 54], [105, 77, 53], [106, 76, 52], [
                107, 75, 51], [108, 74, 50], [109, 73, 49], [110, 72, 49], [111, 71, 49], [112, 70, 49], [113, 69, 49], [114, 69, 49], [115, 68, 48], [116, 67, 47], [117, 66, 46], [118, 66, 46], [119, 66, 46], [120, 66, 46], [121, 65, 46], [122, 65, 46], [123, 65, 46], [124, 64, 46],
            [124, 64, 46], [125, 65, 46], [126, 66, 46], [127, 67, 46], [128, 67, 46], [129, 67, 46], [130, 68, 46], [131, 68, 46], [132, 69, 46], [133, 70, 46], [134, 70, 46], [135, 70, 46], [136, 71, 46], [137, 71, 46], [138, 72, 46], [139, 72, 46], [140, 73, 46], [141, 73, 46], [142, 73, 46], [143, 73, 46], [144, 73, 46], [145, 73, 46], [146, 73, 46], [147, 73, 46], [148, 73, 46], [149, 73, 46], [150, 73, 46], [151, 73, 46], [152, 74, 46], [153, 75, 46], [154, 76, 46], [155, 77, 46], [156, 78, 46], [157, 79, 46], [158, 80, 46], [159, 81, 46], [160, 82, 46], [161, 82, 46], [
                162, 83, 46], [163, 83, 46], [164, 84, 46], [165, 84, 46], [166, 84, 46], [167, 85, 46], [168, 85, 46], [169, 86, 46], [170, 86, 46], [171, 87, 46], [172, 87, 46], [173, 88, 46], [174, 88, 46], [175, 89, 46], [176, 89, 46], [177, 90, 46], [178, 90, 46], [179, 90, 46], [180, 91, 46], [181, 91, 46], [182, 92, 46], [183, 92, 46], [184, 93, 46], [185, 93, 46], [186, 93, 46], [187, 94, 46], [188, 94, 46], [189, 95, 46], [190, 95, 46], [191, 96, 46], [192, 96, 46], [193, 97, 46], [194, 97, 46], [195, 98, 46], [196, 98, 46], [197, 99, 45], [198, 99, 45], [199, 99, 45], [200, 100, 44],
            [200, 100, 44], [200, 101, 44], [200, 102, 44], [200, 103, 44], [200, 104, 44], [200, 105, 44], [200, 106, 44], [200, 107, 44], [200, 108, 44], [200, 109, 44], [200, 110, 44], [200, 111, 44], [200, 112, 44], [200, 113, 44], [200, 114, 44], [200, 115, 44], [200, 116, 44], [200, 117, 44], [200, 118, 44], [200, 119, 44], [
                200, 120, 44], [200, 121, 44], [200, 122, 44], [200, 123, 44], [200, 124, 44], [200, 125, 44], [200, 126, 44], [200, 127, 44], [200, 128, 44], [200, 129, 44], [200, 130, 44], [200, 131, 44], [200, 132, 44], [200, 133, 44], [200, 134, 44], [200, 135, 44], [200, 136, 44], [200, 137, 44], [200, 138, 44], [200, 139, 44],
            [200, 140, 44], [200, 141, 44], [200, 142, 44], [200, 143, 44], [200, 144, 44], [200, 145, 44], [200, 146, 44], [200, 147, 44], [200, 148, 44], [200, 149, 44], [200, 150, 44], [200, 151, 44], [200, 152, 44], [200, 153, 44], [200, 154, 44], [200, 155, 44], [200, 156, 44], [200, 157, 44], [200, 158, 44], [200, 159, 44], [
                200, 160, 44], [200, 161, 44], [200, 162, 44], [200, 163, 44], [200, 164, 44], [200, 165, 45], [200, 166, 45], [199, 167, 46], [199, 168, 46], [198, 169, 47], [198, 170, 47], [197, 171, 48], [197, 172, 48], [197, 173, 49], [197, 174, 49], [196, 175, 50], [196, 176, 50], [195, 177, 51], [195, 178, 51], [195, 179, 51], [194, 180, 52]
        ]

        # Apply scaling factor
        scaled_path = [Point((x) * self.scaling_factor, (y) * self.scaling_factor,
                             (z) * self.scaling_factor) for x, y, z in raw_path]
        rospy.loginfo(
            f"Generated path with scaling factor {self.scaling_factor}")
        return scaled_path

    def current_pose_callback(self, data: PoseStamped):
        self.current_pose = data

    def goal_callback(self, msg: PoseStamped):
        if self.goal is None and self.new_goal_pending is False:
            self.goal = msg.pose.position
            rospy.loginfo(f"New goal received: {self.goal}")
        else:
            rospy.loginfo(
                "Received new goal, will update after reaching the next waypoint")
            self.new_goal_pending = True

    def is_close_to_goal(self, goal: Point, threshold: float) -> bool:
        if self.current_pose is None:
            return False
        dx = goal.x - self.current_pose.pose.position.x
        dy = goal.y - self.current_pose.pose.position.y
        dz = goal.z - self.current_pose.pose.position.z
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        return distance < threshold

    def move_along_path(self):
        if self.goal and self.current_index != 0:
            self.move_to_goal_and_return()
        elif self.current_index < len(self.global_path):
            next_waypoint = self.global_path[self.current_index]
            print(next_waypoint)
            print("--------------")
            self.move_to_point(next_waypoint)
            if self.is_close_to_goal(next_waypoint, self.waypoint_threshold):
                self.current_index += 1
        else:
            rospy.loginfo("Path completed")

    def move_to_point(self, point: Point):
        if not self.current_pose:
            return

        target_pose = PoseStamped()
        target_pose.header.frame_id = "world"
        target_pose.pose.position = point
        target_pose.pose.orientation.w = 1.0

        try:
            self.listener.waitForTransform(
                "world", "base_link", rospy.Time(), rospy.Duration(1.0))
            transformed_point = self.listener.transformPose(
                'base_link', target_pose)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Transformation failed: {e}")
            return

        # Calculate errors for each axis
        error_x = transformed_point.pose.position.x
        error_y = transformed_point.pose.position.y
        error_z = transformed_point.pose.position.z

        # Use PID controllers for each axis
        vel_msg = Twist()
        vel_msg.linear.x = max(-self.max_velocity,
                               min(self.max_velocity, self.pid_x.calculate(error_x)))
        vel_msg.linear.y = max(-self.max_velocity,
                               min(self.max_velocity, self.pid_y.calculate(error_y)))
        vel_msg.linear.z = max(-self.max_velocity,
                               min(self.max_velocity, self.pid_z.calculate(error_z)))

        # Publish the velocity command
        self.cmd_vel_pub.publish(vel_msg)

    def publish_path_as_marker(self):
        marker_array = MarkerArray()

        for i, waypoint in enumerate(self.global_path):
            # Create a spherical marker for the waypoint
            waypoint_marker = Marker()
            waypoint_marker.header.frame_id = "world"
            waypoint_marker.header.stamp = rospy.Time.now()
            waypoint_marker.ns = "path"
            waypoint_marker.id = i * 2  # Unique ID for each waypoint marker
            waypoint_marker.type = Marker.SPHERE
            waypoint_marker.action = Marker.ADD

            waypoint_marker.pose.position = waypoint
            waypoint_marker.pose.orientation.w = 1.0

            waypoint_marker.scale.x = 0.02
            waypoint_marker.scale.y = 0.02
            waypoint_marker.scale.z = 0.02

            waypoint_marker.color.r = 0.0
            waypoint_marker.color.g = 1.0
            waypoint_marker.color.b = 0.0
            waypoint_marker.color.a = 1.0

            marker_array.markers.append(waypoint_marker)

            # Create a text marker to display the waypoint index
            text_marker = Marker()
            text_marker.header.frame_id = "world"
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "path_text"
            text_marker.id = i * 2 + 1  # Unique ID for each text marker
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position = waypoint
            text_marker.pose.position.z += 0.0  # Offset text above the waypoint marker

            text_marker.scale.z = 0.6  # Height of the text

            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            # Display the waypoint index (1-based)
            text_marker.text = str(i + 1)

            marker_array.markers.append(text_marker)

        self.path_marker_pub.publish(marker_array)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.move_along_path()
            self.publish_path_as_marker()
            rate.sleep()


if __name__ == '__main__':
    try:
        drone_follower = DronePathFollower()
        drone_follower.run()
    except rospy.ROSInterruptException:
        pass
