#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import PoseStamped, Point, Twist
import tf
from visualization_msgs.msg import Marker, MarkerArray

class DronePathFollower:
    def __init__(self):
        rospy.init_node('drone_path_follower', anonymous=True)

        self.global_path = [
            Point(3, 3, 2),
            Point(2, 4, 2),
            Point(2, 9, 2),
            Point(4.83, 9.56, 2),
            Point(5.43, 5.57, 2),
        ]

        self.current_index = 0
        self.goal = None
        self.new_goal_pending = False  # New flag to store pending goals until reaching a waypoint
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.path_marker_pub = rospy.Publisher('/path_marker', MarkerArray, queue_size=10)
        rospy.Subscriber('/ground_truth_to_tf/pose', PoseStamped, self.current_pose_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)

        self.listener = tf.TransformListener()
        self.current_pose = None

        self.goal_threshold = 0.1
        self.waypoint_threshold = 0.2
        self.max_velocity = 0.3  # Slower velocity

    def current_pose_callback(self, data: PoseStamped):
        self.current_pose = data

    def goal_callback(self, msg: PoseStamped):
        if self.goal is None and self.new_goal_pending is False:
            self.goal = msg.pose.position
            rospy.loginfo(f"New goal received: {self.goal}")
        else:
            rospy.loginfo("Received new goal, but will update after reaching the next waypoint")
            self.new_goal_pending = True  # Store the goal and wait for waypoint to be reached

    def is_close_to_goal(self, goal: Point, threshold: float) -> bool:
        if self.current_pose is None:
            return False

        dx = goal.x - self.current_pose.pose.position.x
        dy = goal.y - self.current_pose.pose.position.y
        dz = goal.z - self.current_pose.pose.position.z
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        return distance < threshold

    def move_along_path(self):
        if self.goal and self.current_index!=0:  # Handle goal if set
            self.move_to_goal_and_return()
        else:  # Continue moving along the path if no goal is active
            if self.current_index < len(self.global_path):
                next_waypoint = self.global_path[self.current_index]
                self.move_to_point(next_waypoint)
                self.current_index+=1
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

        linear_error_x = transformed_point.pose.position.x
        linear_error_y = transformed_point.pose.position.y
        linear_error_z = transformed_point.pose.position.z

        vel_msg = Twist()
        vel_msg.linear.x = max(-self.max_velocity, min(self.max_velocity, linear_error_x))
        vel_msg.linear.y = max(-self.max_velocity, min(self.max_velocity, linear_error_y))
        vel_msg.linear.z = max(-self.max_velocity, min(self.max_velocity, linear_error_z))

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
