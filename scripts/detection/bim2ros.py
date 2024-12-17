#!/usr/bin/python3

import numpy as np
import ifcopenshell
import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
from sensor_msgs import point_cloud2
import tf2_ros
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from datetime import datetime
import time
from grid_message.msg import ElementOcurrences  # Import the custom message
from visualization_msgs.msg import Marker, MarkerArray  # Import MarkerArray
from concurrent.futures import ThreadPoolExecutor  # For parallel processing
import roslib
import os

# Global variable initialization
tf_buffer = None
onedivres = None
grid_stepy = None
grid_stepz = None
loaded_data = None
loaded_data_zeros = None
element_occurrences = {}
occurrence_publisher = None  # Publisher for element occurrences
marker_publisher = None  # Publisher for markers

PACKAGE_NAME = 'bim2ros'

RES = rospy.get_param('resolution', 0.2)
GRID_SIZEX = rospy.get_param('world_sizeX', 20)
GRID_SIZEY = rospy.get_param('world_sizeY', 20)
GRID_SIZEZ = rospy.get_param('world_sizeZ', 4)
THREAD_COUNT = rospy.get_param('threads', 4)  # Number of threads for parallel processing
MARKER_UPDATE_INTERVAL = rospy.get_param('marker_update_interval', 5)  # Publish interval in seconds

last_marker_publish_time = None  # Time of the last published marker array

def load_ifc_model(model_path):
    try:
        model = ifcopenshell.open(model_path)
        return model
    except Exception as e:
        rospy.logerr(f"Failed to load IFC model: {e}")
        return None

def process_chunk(points):
    """
    Process a chunk of points: Transform, find closest objects, and generate markers.
    """
    transformed_points = []
    local_markers = MarkerArray()
    
    for point in points:
        # Transform point to the map frame
        point_stamped = PointStamped()
        point_stamped.header.stamp = rospy.Time.now()
        point_stamped.header.frame_id = "velodyne_base_link"
        point_stamped.point.x = point[0]
        point_stamped.point.y = point[1]
        point_stamped.point.z = point[2]
        try:
            transformed_point_c = tf_buffer.transform(point_stamped, target_frame='map', timeout=rospy.Duration(0.1))
            transformed_points.append((transformed_point_c.point.x, transformed_point_c.point.y, transformed_point_c.point.z))
        except tf2_ros.TransformException:
            continue

    # Process each transformed point
    for point in transformed_points:
        index = point2grid(point[0], point[1], point[2], onedivres, grid_stepy, grid_stepz)
        if loaded_data_zeros[index] == 0 and loaded_data[index]:
            element_value = loaded_data[index]
            element_occurrences[element_value] = element_occurrences.get(element_value, 0) + 1
            loaded_data_zeros[index] = 1

            # Create a marker for this point
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "seen_sites"
            marker.id = index  # Unique ID for each marker
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = point[2]
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = RES
            marker.scale.y = RES
            marker.scale.z = RES
            marker.color.a = 0.8  # Opacity
            marker.color.r = 0.0  # Red
            marker.color.g = 1.0  # Green
            marker.color.b = 0.0  # Blue
            local_markers.markers.append(marker)
    
    save_element_occurrences(os.path.join(get_package_path(PACKAGE_NAME), 'grids/elements_ocurrences.txt'))
    return local_markers

def point2grid(x, y, z, onedivres, grid_stepy, grid_stepz):
    return int(np.floor(x * onedivres) + (np.floor(y * onedivres) * grid_stepy) + (np.floor(z * onedivres) * grid_stepz))

def get_package_path(package_name):
    return roslib.packages.get_pkg_dir(package_name)

def save_element_occurrences(filename):
    try:
        with open(filename, 'a') as f:
            for element_value, count in element_occurrences.items():
                f.write(f"{element_value} -> {count}\n")
            f.write(f"----------------------------\n")
    except Exception as e:
        rospy.logerr(f"Failed to save element occurrences: {e}")

def pointcloud_callback(msg):
    global last_marker_publish_time, iteration_counter

    iteration_counter += 1

    # Read points from PointCloud2
    points = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))

    # Divide points into chunks
    chunk_size = len(points) // THREAD_COUNT
    point_chunks = [points[i:i + chunk_size] for i in range(0, len(points), chunk_size)]

    # Process chunks in parallel
    markers = MarkerArray()
    with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
        results = executor.map(process_chunk, point_chunks)
        for result in results:
            markers.markers.extend(result.markers)
    
    # Publish all markers at once if enough time has passed
    current_time = rospy.Time.now()
    if last_marker_publish_time is None or (current_time - last_marker_publish_time).to_sec() >= MARKER_UPDATE_INTERVAL:
        marker_publisher.publish(markers)
        last_marker_publish_time = current_time

def main():
    global tf_buffer, onedivres, grid_stepy, grid_stepz, loaded_data, loaded_data_zeros, element_occurrences, occurrence_publisher, marker_publisher, iteration_counter

    rospy.init_node('bim2ros')

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    iteration_counter = 0
    element_occurrences = {}

    onedivres = 1 / RES
    grid_stepy = GRID_SIZEX * onedivres
    grid_stepz = (GRID_SIZEX*onedivres) * (GRID_SIZEY*onedivres)

    file = "/home/rva_container/rva_exchange/catkin_ws/src/bim2ros/grids/semantic_grid_ints.npy"
    loaded_data = np.load(file)
    file = "/home/rva_container/rva_exchange/catkin_ws/src/bim2ros/grids/semantic_grid_zeros.npy"
    loaded_data_zeros = np.load(file)

    occurrence_publisher = rospy.Publisher('element_occurrences', ElementOcurrences, queue_size=10)
    marker_publisher = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
    input("Press any key to start...")
    rospy.Subscriber("velodyne_points", PointCloud2, pointcloud_callback, queue_size=10)

    rospy.spin()

if __name__ == "__main__":
    main()
