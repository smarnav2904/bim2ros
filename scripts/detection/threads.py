#!/usr/bin/python3

import os
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import rospy
import roslib

# Global variable initialization
tf_buffer = None
onedivres = None
grid_stepy = None
grid_stepz = None
loaded_data = None
loaded_data_zeros = None
element_occurrences = {}
total_element_occurrences = {}
ratios_and_times = {}
grids_turned_to_one_times = []
processing_times = []
iteration_counter = 0
occurrence_publisher = None  # Publisher for element occurrences

lock = threading.Lock()  # Lock for safe access to shared variables

PACKAGE_NAME = 'bim2ros'

RES = rospy.get_param('resolution', 0.2)
GRID_SIZEX = rospy.get_param('world_sizeX', 20)
GRID_SIZEY = rospy.get_param('world_sizeY', 20)
GRID_SIZEZ = rospy.get_param('world_sizeZ', 4)

PUBLISH_INTERVAL = 5  # Only publish every 2 seconds
last_publish_time = 0  # Track when we last published

def get_package_path(package_name):
    return roslib.packages.get_pkg_dir(package_name)

def load_ifc_model(model_path):
    try:
        model = ifcopenshell.open(model_path)
        return model
    except Exception as e:
        rospy.logerr(f"Failed to load IFC model: {e}")
        return None

def find_closest_objects_to_point_cloud(points):
    global last_publish_time  # Track the last time we published

    def update_element_occurrences(point):
        index = point2grid(int(point[0]*onedivres), int(point[1]*onedivres), int(point[2]*onedivres), grid_stepy, grid_stepz)
        
        if loaded_data_zeros[index] == 0 and loaded_data[index]:
            element_value = loaded_data[index]
            with lock:
                element_occurrences[element_value] = element_occurrences.get(element_value, 0) + 1
                loaded_data_zeros[index] = 1

    # Use ThreadPoolExecutor to parallelize occurrence updates
    with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers for the number of threads
        executor.map(update_element_occurrences, points)

    closest_to_target_ratio = float('inf')
    closest_element_value = None
    target_ratio = 0.3 # Target ratio is 30%

    with lock:
        for element_value, count in element_occurrences.items():
            total_count = total_element_occurrences.get(element_value, 0)
            if total_count > 0:  # Ensure no division by zero
                ratio = count / total_count
                if ratio <= target_ratio:  # Ensure the ratio is not greater than the target
                    difference = target_ratio - ratio
                    if difference < closest_to_target_ratio:
                        closest_to_target_ratio = difference
                        closest_element_value = element_value
                        ratios_and_times[element_value] = (ratio, datetime.now().isoformat(), count, total_count)

    # Calculate current time
    current_time = time.time()

    # Check if we should publish (only if 2 seconds have passed since the last publish)
    if current_time - last_publish_time >= PUBLISH_INTERVAL:
        if closest_element_value is not None:
            closest_ratio, timestamp, count, total_count = ratios_and_times[closest_element_value]
            rospy.loginfo(f"\033[92m[PUBLISH] Value closest to target ratio {closest_element_value}: Ratio {closest_ratio:.2%} at {timestamp} \033[0m")

            # Publish the closest element occurrence
            occurrence_msg = ElementOcurrences()
            occurrence_msg.element_value = int(closest_element_value)
            occurrence_msg.ratio = closest_ratio * 100
            occurrence_publisher.publish(occurrence_msg)

            # Update last publish time
            last_publish_time = current_time
        else:
            rospy.loginfo(f"\033[93m[PUBLISH SKIPPED] No valid element to publish \033[0m")
    else:
        # Show how much time is left before the next publish
        time_left = PUBLISH_INTERVAL - (current_time - last_publish_time)
        rospy.loginfo(f"\033[93mNext publish in {time_left:.2f} seconds\033[0m")

    rospy.loginfo("-------------------------------------------------------")
    calculate_grid_zeros_ratio()

def calculate_grid_zeros_ratio():
    total_counts = sum(total_element_occurrences.values())
    grids_turned_to_one = sum(element_occurrences.values())
    ratio = grids_turned_to_one / total_counts
    timestamp = datetime.now().isoformat()
    grids_turned_to_one_times.append((grids_turned_to_one, total_counts, ratio, timestamp))
    rospy.loginfo(f"\033[94mGrids explored: {grids_turned_to_one} / {total_counts} ({ratio:.2%}) \033[0m")

def point2grid(x, y, z, grid_stepy, grid_stepz ):
    return int(x) + int((y) * grid_stepy) + int((z) * (grid_stepz))

def save_element_occurrences(filename):
    try:
        with open(filename, 'a') as f:
            for element_value, (ratio, timestamp,count, total_count) in ratios_and_times.items():
                f.write(f"{element_value} -> {count} / {total_count} ({ratio:.2%})\n")
            f.write("-------------------------------------------------------\n")
            # for grids_turned_to_one, total_counts, ratio, timestamp in grids_turned_to_one_times:
            #     f.write(f"Grids turned to 1: {grids_turned_to_one} / {total_counts} ({ratio:.2%})\n")
    except Exception as e:
        rospy.logerr(f"Failed to save element occurrences: {e}")

def pointcloud_callback(msg):
    global iteration_counter

    iteration_counter += 1
    points = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    pcl_example = [(point[0], point[1], point[2]) for point in points]

    rospy.loginfo(f"\033[93mIteration {iteration_counter}\033[0m")

    transformed_points = []

    def transform_point(point):
        point_stamped = PointStamped()
        point_stamped.header.stamp = rospy.Time.now()
        point_stamped.header.frame_id = "velodyne_base_link"
        point_stamped.point.x = point[0]
        point_stamped.point.y = point[1]
        point_stamped.point.z = point[2]
        try:
            transformed_point_c = tf_buffer.transform(point_stamped, target_frame='map', timeout=rospy.Duration(0.1))
            return (transformed_point_c.point.x, transformed_point_c.point.y, transformed_point_c.point.z)
        except tf2_ros.TransformException as ex:
            rospy.logwarn(f"Transform failed: {ex}")
            return None

    # Use ThreadPoolExecutor to parallelize point transformations
    with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers for the number of threads
        futures = [executor.submit(transform_point, point) for point in pcl_example]
        for future in as_completed(futures):
            result = future.result()
            if result:
                transformed_points.append(result)

    find_closest_objects_to_point_cloud(transformed_points)
    save_element_occurrences(os.path.join(get_package_path(PACKAGE_NAME), 'grids/elements_ocurrences.txt'))

def calculate_total_occurrences():
    unique, counts = np.unique(loaded_data, return_counts=True)
    for value, count in zip(unique, counts):
        if value != 0:  # Assuming 0 is not a valid element value to track
            total_element_occurrences[value] = count

def main():
    global tf_buffer, onedivres, grid_stepy, grid_stepz, loaded_data, loaded_data_zeros, element_occurrences, occurrence_publisher

    rospy.init_node('bim2ros')

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    element_occurrences = {}

    onedivres = 1 / RES

    tam_x = int(GRID_SIZEX * onedivres)
    tam_y = int(GRID_SIZEY * onedivres)
    tam_z = int(GRID_SIZEZ * onedivres)

    size = tam_x * tam_y * tam_z

    grid_stepy = tam_x
    grid_stepz = tam_x * tam_y

    file = os.path.join(get_package_path(PACKAGE_NAME), 'grids/semantic_grid_ints.npy')
    loaded_data = np.load(file)
    file = os.path.join(get_package_path(PACKAGE_NAME), 'grids/semantic_grid_zeros.npy')
    loaded_data_zeros = np.load(file)

    calculate_total_occurrences()

    occurrence_publisher = rospy.Publisher('element_occurrences', ElementOcurrences, queue_size=10)

    rospy.Subscriber("velodyne_points", PointCloud2, pointcloud_callback)

    rospy.spin()

if __name__ == "__main__":
    main()
