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

RES = 0.2
GRID_SIZEX = 220
GRID_SIZEY = 220
GRID_SIZEZ = 20

def load_ifc_model(model_path):
    try:
        model = ifcopenshell.open(model_path)
        return model
    except Exception as e:
        rospy.logerr(f"Failed to load IFC model: {e}")
        return None

def find_closest_objects_to_point_cloud(points):

    
    for point in points:
        index = point2grid(point[0], point[1], point[2], onedivres, grid_stepy, grid_stepz)
        if loaded_data_zeros[index] == 0 and loaded_data[index]:
            element_value = loaded_data[index]
            element_occurrences[element_value] = element_occurrences.get(element_value, 0) + 1
            loaded_data_zeros[index] = 1
        

    closest_to_target_ratio = float('inf')
    closest_element_value = None
    target_ratio = 0.30  # Target ratio is 30%

    for element_value, count in element_occurrences.items():
        total_count = total_element_occurrences.get(element_value, 0)
        if total_count > 0:  # Ensure no division by zero
            ratio = count / total_count
            if ratio <= target_ratio:  # Ensure the ratio is not greater than the target
                difference = target_ratio - ratio
                if difference < closest_to_target_ratio:
                    closest_to_target_ratio = difference
                    closest_element_value = element_value
                    ratios_and_times[element_value] = (ratio, datetime.now().isoformat())

    if closest_element_value is not None:
        closest_ratio, timestamp = ratios_and_times[closest_element_value]
        rospy.loginfo(f"\033[92mValue closest to target ratio {closest_element_value}: Ratio {closest_ratio:.2%} at {timestamp} \033[0m")

        # Publish the closest element occurrence
        occurrence_msg = ElementOcurrences()
        occurrence_msg.element_value = int(closest_element_value)
        occurrence_msg.ratio = closest_ratio * 100
        occurrence_publisher.publish(occurrence_msg)

    rospy.loginfo("-------------------------------------------------------")
    calculate_grid_zeros_ratio()

def calculate_grid_zeros_ratio():
    total_counts = sum(total_element_occurrences.values())
    grids_turned_to_one = sum(element_occurrences.values())
    ratio = grids_turned_to_one / total_counts
    timestamp = datetime.now().isoformat()
    grids_turned_to_one_times.append((grids_turned_to_one, total_counts, ratio, timestamp))
    rospy.loginfo(f"\033[94mGrids explored: {grids_turned_to_one} / {total_counts} ({ratio:.2%}) \033[0m")

def point2grid(x, y, z, onedivres, grid_stepy, grid_stepz):
    
    return int(np.floor(x * onedivres) + (np.floor(y * onedivres) * grid_stepy) + (np.floor(z * onedivres) * grid_stepz))
    

def crop_cloud(cl, maxdist, mindist=0.00):
    cldist = np.linalg.norm(cl[:, 0:2], axis=1)
    return cl[(cldist > mindist) & (cldist <= maxdist), :]

def save_element_occurrences(filename):
    try:
        with open(filename, 'w') as f:
            for element_value, (ratio, timestamp) in ratios_and_times.items():
                f.write(f"{element_value}:{ratio:.2%}\n")
            f.write("-------------------------------------------------------\n")
            for grids_turned_to_one, total_counts, ratio, timestamp in grids_turned_to_one_times:
                f.write(f"Grids turned to 1: {grids_turned_to_one} / {total_counts} ({ratio:.2%})\n")
    except Exception as e:
        rospy.logerr(f"Failed to save element occurrences: {e}")

def pointcloud_callback(msg):
    global iteration_counter
    iteration_counter += 1

    points = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    pcl_example = [(point[0], point[1], point[2]) for point in points]

    max_dist = 6
    pcl_example = crop_cloud(np.array(pcl_example), max_dist)
    rospy.loginfo(f"\033[93mIteration {iteration_counter}: Distance reduced to: {max_dist} \033[0m")
    transformed_points = []

    for point in pcl_example:
        point_stamped = PointStamped()
        point_stamped.header.stamp = rospy.Time.now()
        point_stamped.header.frame_id = "velodyne_base_link"
        point_stamped.point.x = point[0]
        point_stamped.point.y = point[1]
        point_stamped.point.z = point[2]
        try:
            transformed_point_c = tf_buffer.transform(point_stamped, target_frame='map', timeout=rospy.Duration(0.1))
            transformed_points.append((transformed_point_c.point.x, transformed_point_c.point.y, transformed_point_c.point.z))
        except tf2_ros.TransformException as ex:
            rospy.logwarn(f"Transform failed: {ex}")


    find_closest_objects_to_point_cloud(transformed_points)
    save_element_occurrences("element_occurrences.txt")
    # rospy.loginfo(f"\033[93mIteration {iteration_counter}: Elapsed time for this iteration: {processing_time:.2f} seconds \033[0m")
    # rospy.loginfo("\033[93m---------------------------------------\033[0m")

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
    grid_stepy = GRID_SIZEX * onedivres
    grid_stepz = (GRID_SIZEX*onedivres) * (GRID_SIZEY*onedivres)

    file = "/home/rva_container/rva_exchange/catkin_ws/src/bim2ros/scripts/semantic_grid_ints.npy"
    loaded_data = np.load(file)
    file = "/home/rva_container/rva_exchange/catkin_ws/src/bim2ros/scripts/semantic_grid_zeros.npy"
    loaded_data_zeros = np.load(file)

    calculate_total_occurrences()

    occurrence_publisher = rospy.Publisher('element_occurrences', ElementOcurrences, queue_size=10)

    rospy.Subscriber("velodyne_points", PointCloud2, pointcloud_callback)

    rospy.spin()

if __name__ == "__main__":
    main()
