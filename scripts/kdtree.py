#!/usr/bin/python3

import numpy as np
import ifcopenshell
import ifcopenshell.geom
import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
from sensor_msgs import point_cloud2
import tf2_ros
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
import time
import multiprocessing
from grid_message.msg import ElementOcurrences
import csv

class BIM2ROSProcessor:
    def __init__(self, model_path, resolution, grid_size_x, grid_size_y, grid_size_z):
        self.resolution = resolution
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.grid_size_z = grid_size_z
        self.onedivres = 1 / resolution
        self.grid_stepy = grid_size_x * self.onedivres
        self.grid_stepz = (grid_size_x * self.onedivres) * (grid_size_y * self.onedivres)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.occurrence_publisher = rospy.Publisher('element_occurrences', ElementOcurrences, queue_size=10)
        self.element_occurrences = {}
        self.tree = self.initialize_ifc_model(model_path)
        self.total_processing_time = 0.0
        self.iteration_counter = 0
        self.metrics = []  # To store metrics for each iteration
    
    def initialize_ifc_model(self, model_path):
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

    def pointcloud_callback(self, msg):
        self.iteration_counter += 1
        points = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pcl_example = [(point[0], point[1], point[2]) for point in points]
        transformed_points = []
        num_points = len(pcl_example)

        for point in pcl_example:
            point_stamped = PointStamped()
            point_stamped.header.stamp = rospy.Time.now()
            point_stamped.header.frame_id = "velodyne_base_link"
            point_stamped.point.x = point[0]
            point_stamped.point.y = point[1]
            point_stamped.point.z = point[2]
            try:
                transformed_point_c = self.tf_buffer.transform(point_stamped, target_frame='map', timeout=rospy.Duration(0.1))
                transformed_points.append((transformed_point_c.point.x, transformed_point_c.point.y, transformed_point_c.point.z))
            except tf2_ros.TransformException as ex:
                rospy.logwarn(f"Transform failed: {ex}")

        start = time.time()
        num_elements = self.find_objects(transformed_points)
        end = time.time()

        processing_time = end - start
        self.total_processing_time += processing_time
        average_time = self.total_processing_time / self.iteration_counter
        average_time_per_point = average_time / num_points if num_points > 0 else 0.0
        ratio_points_with_element = num_elements / num_points if num_points > 0 else 0.0

        # Store the metrics for this iteration
        self.metrics.append({
            "Iteration": self.iteration_counter,
            "Number of Points": num_points,
            "Processing Time (s)": processing_time,
            "Average Time (s)": average_time,
            "Average Time per Point (s)": average_time_per_point,
            "Points with Element": num_elements,
            "Ratio": ratio_points_with_element
        })

        # Print the metrics for the current iteration
        print(f"Iteration: {self.iteration_counter}, Number of Points: {num_points}, "
              f"Processing Time: {processing_time:.4f}s, Average Time: {average_time:.4f}s, "
              f"Average Time per Point: {average_time_per_point:.8f}s, "
              f"Points with Element: {num_elements}, Ratio: {ratio_points_with_element:.4f}")

        # Save to CSV after 100 iterations
        if self.iteration_counter == 100:
            self.save_metrics_to_csv('metrics.csv')
            rospy.loginfo("Saved metrics to metrics.csv")

    def find_objects(self, points):
        count = 0
        for p in points:
            elements = self.tree.select(p, extend=0.05)
            if elements:
                count += 1
            
        return count

    def save_metrics_to_csv(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ["Iteration", "Number of Points", "Processing Time (s)", "Average Time (s)",
                          "Average Time per Point (s)", "Points with Element", "Ratio"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for data in self.metrics:
                writer.writerow(data)

def main():
    rospy.init_node('bim2ros')

    model_path = '/home/rva_container/rva_exchange/catkin_ws/src/bim2ros/models/model.ifc'
    resolution = 0.1
    grid_size_x = 30
    grid_size_y = 30
    grid_size_z = 30

    processor = BIM2ROSProcessor(model_path, resolution, grid_size_x, grid_size_y, grid_size_z)

    rospy.Subscriber("velodyne_points", PointCloud2, processor.pointcloud_callback)

    rospy.spin()

if __name__ == "__main__":
    main()
