#!/usr/bin/env python

import multiprocessing
import ifcopenshell
import ifcopenshell.geom
import json
import rospy
import numpy as np
import math
import time
from grid_message.msg import ElementOcurrences
from geometry_msgs.msg import PoseStamped, Twist
from visualization_msgs.msg import Marker

def setup_ifc_geometry(ifc_file_path):
    ifc_file = ifcopenshell.open(ifc_file_path)
    settings = ifcopenshell.geom.settings()
    settings.set(settings.DISABLE_OPENING_SUBTRACTIONS, False)
    settings.set(settings.USE_WORLD_COORDS, True)
    return ifc_file, settings

def initialize_iterator(settings, ifc_file):
    iterator = ifcopenshell.geom.iterator(settings, ifc_file, multiprocessing.cpu_count())
    tree = ifcopenshell.geom.tree()
    if iterator.initialize():
        while True:
            shape = iterator.get_native()
            tree.add_element(shape)
            if not iterator.next():
                break
    return tree

def load_guid_mapping(json_file_path):
    with open(json_file_path, 'r') as f:
        return {int(v): k for k, v in json.load(f).items()}

def get_element_by_guid(ifc_file, guid):
    return ifc_file.by_guid(guid)

def calculate_projections(element_pos, drone_pos):
    element_pos = np.array(element_pos)
    drone_pos = np.array(drone_pos)
    
    x_vector = np.array([1, 0, 0])
    y_vector = np.array([0, 1, 0])
    z_vector = np.array([0, 0, 1])

    proj_x = element_pos + np.dot(drone_pos - element_pos, x_vector) * x_vector
    proj_y = element_pos + np.dot(drone_pos - element_pos, y_vector) * y_vector
    proj_z = element_pos + np.dot(drone_pos - element_pos, z_vector) * z_vector

    return proj_x, proj_y, proj_z

def create_marker(position):
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "projection"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    return marker

def callback_element(msg):
    value = msg.element_value
    ratio = msg.ratio * 100
    guid = guid_mapping.get(value)

    # rospy.loginfo(f"Value: {value}")
    # rospy.loginfo(f"GUID: {guid}")
    if guid:
        element = get_element_by_guid(ifc_file, guid)
        if element:
            shape = ifcopenshell.geom.create_shape(settings, element)

            verts = shape.geometry.verts
            vertX = np.mean(verts[0::3])
            vertY = np.mean(verts[1::3])
            vertZ = np.mean(verts[2::3])
            pos = [vertX, vertY, vertZ]

            # rospy.loginfo(f"Element found: {pos}")

            global current_element_pos
            current_element_pos = pos
        else:
            rospy.logwarn(f"No element found for GUID: {guid}")
    else:
        rospy.logwarn(f"No GUID found for value: {value}")

def callback_drone(msg):
    global drone_pos
    drone_pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

    if current_element_pos:
        proj_x, proj_y, proj_z = calculate_projections(current_element_pos, drone_pos)

        dists = {
            'X': (proj_x, np.linalg.norm(proj_x - drone_pos)),
            'Y': (proj_y, np.linalg.norm(proj_y - drone_pos)),
            'Z': (proj_z, np.linalg.norm(proj_z - drone_pos))
        }

        closest_axis = min(dists, key=lambda k: dists[k][1])
        closest_projection = dists[closest_axis][0]
        # rospy.loginfo(f"Closest projection to drone position: {closest_projection}")

        marker = create_marker(closest_projection)
        marker_pub.publish(marker)
        
        # Move the drone towards the marker
        move_to_target(closest_projection)
    else:
        rospy.logwarn("No current element position to calculate projections from.")

def move_to_target(target_pose):
    
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(10)  # Publish rate (10 Hz)

    while not rospy.is_shutdown():
        if drone_pos is not None:
            # Calculate differences in three dimensions
            dx = target_pose[0] - drone_pos[0]
            dy = target_pose[1] - drone_pos[1]
            dz = target_pose[2] - drone_pos[2]

            # Threshold to check if drone has reached the target position
            if abs(dx) < 0.3 and abs(dy) < 0.3 and abs(dz) < 0.3:
                rospy.loginfo(f"Target reached at {target_pose}!")
                stop_drone(cmd_vel_pub)
                time.sleep(2)  # Pause for 2 seconds
                break

            # Calculate direction and normalize velocity to have a constant magnitude
            distance = math.sqrt(dx**2 + dy**2 + dz**2)
            vel_msg = Twist()
            vel_msg.linear.x = (dx / distance) * 0.1
            vel_msg.linear.y = (dy / distance) * 0.1
            vel_msg.linear.z = (dz / distance) * 0.1
            vel_msg.angular.x = 0.0
            vel_msg.angular.y = 0.0
            vel_msg.angular.z = 0.0

            # Publish the velocity on the /cmd_vel topic
            cmd_vel_pub.publish(vel_msg)

        rate.sleep()

def stop_drone(cmd_vel_pub):
    stop_msg = Twist()
    stop_msg.linear.x = 0.0
    stop_msg.linear.y = 0.0
    stop_msg.linear.z = 0.0
    stop_msg.angular.x = 0.0
    stop_msg.angular.y = 0.0
    stop_msg.angular.z = 0.0
    cmd_vel_pub.publish(stop_msg)

if __name__ == "__main__":
    rospy.init_node('ifc_ros_node')

    # Paths
    ifc_file_path = '/home/rva_container/rva_exchange/catkin_ws/src/bim2ros/models/model.ifc'
    json_file_path = '/home/rva_container/rva_exchange/catkin_ws/src/bim2ros/scripts/global_id_mapping.json'

    # Setup IFC geometry and load GUID mapping
    ifc_file, settings = setup_ifc_geometry(ifc_file_path)
    tree = initialize_iterator(settings, ifc_file)
    guid_mapping = load_guid_mapping(json_file_path)

    # Initialize global variable for current element position
    current_element_pos = None

    # Initialize marker publisher
    marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)

    # Initialize cmd_vel publisher and subscribe to ROS topics
    rospy.Subscriber('element_occurrences', ElementOcurrences, callback_element)
    rospy.Subscriber('ground_truth_to_tf/pose', PoseStamped, callback_drone)

    rospy.spin()
