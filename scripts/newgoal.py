#!/usr/bin/env python

import multiprocessing
import ifcopenshell
import ifcopenshell.geom
import json
import rospy
import numpy as np
from grid_message.msg import ElementOcurrences
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import os

RES = 0.2
GRID_SIZEX = 220
GRID_SIZEY = 220
GRID_SIZEZ = 20

# Global variable for drone position
drone_position = [0, 0, 0]  # Initialize with a default position

def setup_ifc_geometry(ifc_file_path):
    try:
        ifc_file = ifcopenshell.open(ifc_file_path)
        settings = ifcopenshell.geom.settings()
        settings.set(settings.DISABLE_OPENING_SUBTRACTIONS, False)
        settings.set(settings.USE_WORLD_COORDS, True)
        return ifc_file, settings
    except Exception as e:
        rospy.logerr(f"Failed to set up IFC geometry: {e}")
        rospy.signal_shutdown("IFC Setup Failed")
        return None, None

def initialize_iterator(settings, ifc_file):
    try:
        iterator = ifcopenshell.geom.iterator(settings, ifc_file, multiprocessing.cpu_count())
        tree = ifcopenshell.geom.tree()
        if iterator.initialize():
            while True:
                shape = iterator.get_native()
                tree.add_element(shape)
                if not iterator.next():
                    break
        return tree
    except Exception as e:
        rospy.logerr(f"Failed to initialize IFC iterator: {e}")
        rospy.signal_shutdown("IFC Iterator Initialization Failed")
        return None

def load_guid_mapping(json_file_path):
    try:
        with open(json_file_path, 'r') as f:
            return {int(v): k for k, v in json.load(f).items()}
    except Exception as e:
        rospy.logerr(f"Failed to load GUID mapping: {e}")
        rospy.signal_shutdown("GUID Mapping Loading Failed")
        return None

def get_element_by_guid(ifc_file, guid):
    try:
        return ifc_file.by_guid(guid)
    except Exception as e:
        rospy.logerr(f"Failed to get element by GUID: {e}")
        return None

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

def point2grid(x, y, z):
    return int(np.floor(x * onedivres) + 
               (np.floor(y * onedivres) * grid_stepy) + 
               (np.floor(z * onedivres) * grid_stepz))

def check_if_cells_free(point, r):
    # Convert radius from meters to number of cells
    
    # Calculate neighboring points within the N-cell radius around the given point
    for dx in np.arange(-r, r, RES):
        for dy in np.arange(-r, r, RES):
            for dz in np.arange(-r, r, RES):
                neighbor_point = [point[0] + dx, point[1] + dy, point[2] + dz]
                index = point2grid(neighbor_point[0], neighbor_point[1], neighbor_point[2])
                if loaded_data[index]:  # If any neighbor is occupied
                    # return False
                    print('')
    return True  # All neighbors are free


def move_drone_to_position(position):
    try:
        goal = PoseStamped()
        goal.header = Header(stamp=rospy.Time.now(), frame_id="map")
        goal.pose.position.x = position[0]
        goal.pose.position.y = position[1]
        goal.pose.position.z = position[2]
        goal.pose.orientation.x = 0
        goal.pose.orientation.y = 0
        goal.pose.orientation.z = 0
        goal.pose.orientation.w = 1
        goal_publisher.publish(goal)
    except Exception as e:
        rospy.logerr(f"Failed to move drone to position: {e}")

def callback_drone_position(msg):
    global drone_position
    try:
        drone_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
    except Exception as e:
        rospy.logerr(f"Error in drone position callback: {e}")

def callback_element(msg):
    global drone_position
    try:
        value = msg.element_value
        guid = guid_mapping.get(value)

        if guid:
            element = get_element_by_guid(ifc_file, guid)
            if element:
                shape = ifcopenshell.geom.create_shape(settings, element)
                verts = shape.geometry.verts
                vertX = np.mean(verts[0::3])
                vertY = np.mean(verts[1::3])
                vertZ = np.mean(verts[2::3])
                pos = [vertX, vertY, vertZ]

                # Calculate projections
                proj_x, proj_y, proj_z = calculate_projections(pos, drone_position)

                # Calculate distances to the projections
                dist_proj_x = np.linalg.norm(drone_position - proj_x)
                dist_proj_y = np.linalg.norm(drone_position - proj_y)
                dist_proj_z = np.linalg.norm(drone_position - proj_z)

                # Determine the closest projection
                min_dist = min(dist_proj_x, dist_proj_y, dist_proj_z)

                thresh = 1.5
                # Calculate the target position 1.5 meters away from the element center along the closest axis
                if min_dist == dist_proj_x:
                    point = pos + thresh * np.array([1, 0, 0])
                elif min_dist == dist_proj_y:
                    point = pos + thresh * np.array([0, 1, 0])
                else:
                    point = pos + thresh * np.array([0, 0, 1])

                radius = 1.0 #Add in meters!!

                # Check if the point and its neighbors are free
                if check_if_cells_free(point, radius):
                    rospy.loginfo(f"FREE: {point}")
                    move_drone_to_position(point)
                else:
                    rospy.loginfo(f"OCCUPIED or neighboring cells occupied")

            else:
                rospy.logwarn(f"No element found for GUID: {guid}")
        else:
            rospy.logwarn(f"No GUID found for value: {value}")
    except Exception as e:
        rospy.logerr(f"Error in element callback: {e}")

if __name__ == "__main__":
    rospy.init_node('ifc_ros_node')
    global onedivres, grid_stepy, grid_stepz

    ifc_file_path = '/home/rva_container/rva_exchange/catkin_ws/src/bim2ros/models/model.ifc'
    json_file_path = '/home/rva_container/rva_exchange/catkin_ws/src/bim2ros/scripts/global_id_mapping.json'

    file = "/home/rva_container/rva_exchange/catkin_ws/src/bim2ros/scripts/semantic_grid_ints.npy"
    loaded_data = np.load(file)

    onedivres = 1 / RES
    grid_stepy = GRID_SIZEX * onedivres
    grid_stepz = (GRID_SIZEX * onedivres) * (GRID_SIZEY * onedivres)

    ifc_file, settings = setup_ifc_geometry(ifc_file_path)
    if not ifc_file or not settings:
        rospy.signal_shutdown("Failed to initialize IFC and settings")
        exit(1)

    tree = initialize_iterator(settings, ifc_file)
    guid_mapping = load_guid_mapping(json_file_path)
    if not guid_mapping:
        rospy.signal_shutdown("Failed to load GUID mapping")
        exit(1)

    goal_publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
    rospy.Subscriber('element_occurrences', ElementOcurrences, callback_element)
    rospy.Subscriber('/ground_truth_to_tf/pose', PoseStamped, callback_drone_position)

    rospy.spin()
