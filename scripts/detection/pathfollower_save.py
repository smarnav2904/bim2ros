#!/usr/bin/env python3

import rospy
import math
import time
from geometry_msgs.msg import PoseStamped, Point, Twist
from nav_msgs.msg import Path
import tf
from visualization_msgs.msg import Marker, MarkerArray

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
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)

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
        self.global_path = []
        self.current_index = 0
        self.goal = None
        self.new_goal_pending = False
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.path_marker_pub = rospy.Publisher('/path_marker', MarkerArray, queue_size=10)
        
        # Subscribe to the required topics
        rospy.Subscriber('/tsp_path', Path, self.tsp_path_callback)
        rospy.Subscriber('/ground_truth_to_tf/pose', PoseStamped, self.current_pose_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)

        # Initialize other required variables
        self.listener = tf.TransformListener()
        self.current_pose = None
        self.goal_threshold = 0.1
        self.waypoint_threshold = 0.2
        self.max_velocity = 0.2
        self.path_received = False

    def tsp_path_callback(self, msg: Path):
        if not self.path_received:
            self.global_path = [pose.pose.position for pose in msg.poses]
            self.current_index = 0
            self.path_received = True
            rospy.loginfo("Path received with {} waypoints, path set once".format(len(self.global_path)))

    def current_pose_callback(self, data: PoseStamped):
        self.current_pose = data

    def goal_callback(self, msg: PoseStamped):
        if self.goal is None and self.new_goal_pending is False:
            self.goal = msg.pose.position
            rospy.loginfo(f"New goal received: {self.goal}")
        else:
            rospy.loginfo("Received new goal, will update after reaching the next waypoint")
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
            self.listener.waitForTransform("world", "base_link", rospy.Time(), rospy.Duration(1.0))
            transformed_point = self.listener.transformPose('base_link', target_pose)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Transformation failed: {e}")
            return

        # Calculate errors for each axis
        error_x = transformed_point.pose.position.x
        error_y = transformed_point.pose.position.y
        error_z = transformed_point.pose.position.z

        # Use PID controllers for each axis
        vel_msg = Twist()
        vel_msg.linear.x = max(-self.max_velocity, min(self.max_velocity, self.pid_x.calculate(error_x)))
        vel_msg.linear.y = max(-self.max_velocity, min(self.max_velocity, self.pid_y.calculate(error_y)))
        vel_msg.linear.z = max(-self.max_velocity, min(self.max_velocity, self.pid_z.calculate(error_z)))

        # Publish the velocity command
        self.cmd_vel_pub.publish(vel_msg)

    def publish_path_as_marker(self):
        marker_array = MarkerArray()

        for i, waypoint in enumerate(self.global_path):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "path"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position = waypoint
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

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
