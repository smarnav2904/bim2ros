#!/usr/bin/python3

import os
import numpy as np
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element
import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
from sensor_msgs import point_cloud2
import tf2_ros
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from datetime import datetime
import time
from grid_message.msg import ElementOcurrences
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import csv
import multiprocessing
import json

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
metrics = []  # To store metrics for each iteration
lock = threading.Lock()  # Lock for safe access to shared variables

RES = 0.2
GRID_SIZEX = 30
GRID_SIZEY = 30
GRID_SIZEZ = 30

def load_ifc_model(model_path):
    ifc_file = ifcopenshell.open(model_path)
    settings = ifcopenshell.geom.settings()
    settings.set(settings.DISABLE_OPENING_SUBTRACTIONS, False)
    settings.set(settings.USE_WORLD_COORDS, True)
    iterator = ifcopenshell.geom.iterator(settings, ifc_file, multiprocessing.cpu_count())
    tree = ifcopenshell.geom.tree()
    if iterator.initialize():
        while True:
            shape = iterator.get_native()
            tree.add_element(shape)
            if not iterator.next():
                break
    return tree
    
        

def find_closest_objects_to_point_cloud(points):

    count = 0
    for point in points:
        index = point2grid(point[0], point[1], point[2], onedivres, grid_stepy, grid_stepz)

        if loaded_data[index]:
            element_value = loaded_data[index]
            count += 1
            
    return count
    
def point2grid(x, y, z, onedivres, grid_stepy, grid_stepz):
    return int(np.floor(x * onedivres) + (np.floor(y * onedivres) * grid_stepy) + (np.floor(z * onedivres) * grid_stepz))


def pointcloud_callback(msg):
    global iteration_counter
    iteration_counter += 1

    points = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    pcl_example = [(point[0], point[1], point[2]) for point in points]
    num_points = len(pcl_example)

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
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(transform_point, point) for point in pcl_example]
        for future in as_completed(futures):
            result = future.result()
            if result:
                transformed_points.append(result)

    start = time.time()
    num_elements = find_closest_objects_to_point_cloud(transformed_points)
    end = time.time()

    processing_time = end - start
    total_processing_time = sum([m["Processing Time (s)"] for m in metrics]) + processing_time
    average_time = total_processing_time / iteration_counter
    average_time_per_point = average_time / num_points if num_points > 0 else 0.0
    ratio_points_with_element = num_elements / num_points if num_points > 0 else 0.0

    # Store the metrics for this iteration
    metrics.append({
        "Iteration": iteration_counter,
        "Number of Points": num_points,
        "Processing Time (s)": processing_time,
        "Average Time (s)": average_time,
        "Average Time per Point (s)": average_time_per_point,
        "Points with Element": num_elements,
        "Ratio": ratio_points_with_element
    })

    # Print the metrics for the current iteration
    rospy.loginfo(f"Iteration: {iteration_counter}, Number of Points: {num_points}, "
                  f"Processing Time: {processing_time:.4f}s, Average Time: {average_time:.4f}s, "
                  f"Average Time per Point: {average_time_per_point:.8f}s, "
                  f"Points with Element: {num_elements}, Ratio: {ratio_points_with_element:.4f}")

    # Save to CSV after 100 iterations
    if iteration_counter == 100:
        save_metrics_to_csv('metrics.csv')
        rospy.loginfo("Saved metrics to metrics.csv")

def save_metrics_to_csv(filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ["Iteration", "Number of Points", "Processing Time (s)", "Average Time (s)",
                      "Average Time per Point (s)", "Points with Element", "Ratio"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for data in metrics:
            writer.writerow(data)

def load_json(path):
    with open(path, 'r') as archivo:
        data = json.load(archivo)
    
    return data
def main():
    global tf_buffer, onedivres, grid_stepy, grid_stepz, loaded_data, loaded_data_zeros, element_occurrences, occurrence_publisher, tree, global_id_mapping

    rospy.init_node('bim2ros')

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    global_id_mapping = load_json('/home/rva_container/rva_exchange/catkin_ws/src/bim2ros/scripts/global_id_mapping.json')

    tree = load_ifc_model('/home/rva_container/rva_exchange/catkin_ws/src/bim2ros/models/model.ifc')
    onedivres = 1 / RES
    grid_stepy = GRID_SIZEX * onedivres
    grid_stepz = (GRID_SIZEX * onedivres) * (GRID_SIZEY * onedivres)

    file = "/home/rva_container/rva_exchange/catkin_ws/src/bim2ros/scripts/semantic_grid_ints.npy"
    loaded_data = np.load(file)
    file = "/home/rva_container/rva_exchange/catkin_ws/src/bim2ros/scripts/semantic_grid_zeros.npy"
    loaded_data_zeros = np.load(file)


    occurrence_publisher = rospy.Publisher('element_occurrences', ElementOcurrences, queue_size=10)

    rospy.Subscriber("velodyne_points", PointCloud2, pointcloud_callback)

    rospy.spin()

if __name__ == "__main__":
    main()
