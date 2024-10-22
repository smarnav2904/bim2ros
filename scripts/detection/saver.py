#!/usr/bin/env python

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import tf
import numpy as np

def pointcloud_callback(msg, tf_listener):
    try:
        # Wait for the transform from velodyne_base_link to map
        tf_listener.waitForTransform('/map', '/velodyne_base_link', rospy.Time(0), rospy.Duration(4.0))

        # Get the transformation between velodyne_base_link and map
        (trans, rot) = tf_listener.lookupTransform('/map', '/velodyne_base_link', rospy.Time(0))
        translation_matrix = tf.transformations.translation_matrix(trans)
        rotation_matrix = tf.transformations.quaternion_matrix(rot)

        # Combine the translation and rotation into a single transformation matrix
        transform_matrix = np.dot(translation_matrix, rotation_matrix)

        # Open file to save the transformed points
        with open("/home/rva_container/rva_exchange/catkin_ws/velodyne_points_xyz_map_frame.txt", "a") as f:
            # Iterate through the PointCloud2 message using read_points
            for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                # Unpack point coordinates (x, y, z)
                point_in_base_link = np.array([point[0], point[1], point[2], 1.0])  # Homogeneous coordinates

                # Transform point to map frame
                point_in_map_frame = np.dot(transform_matrix, point_in_base_link)

                # Write the transformed point to the file in XYZ format
                f.write(f"{point_in_map_frame[0]:.4f} {point_in_map_frame[1]:.4f} {point_in_map_frame[2]:.4f}\n")

        rospy.loginfo("Saved transformed points to /tmp/velodyne_points_xyz_map_frame.txt")

    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logerr("TF exception: %s", e)

def main():
    # Initialize ROS node
    rospy.init_node('velodyne_pointcloud_saver_with_tf_no_ros_numpy', anonymous=True)

    # Create a TF listener
    tf_listener = tf.TransformListener()

    # Subscribe to Velodyne PointCloud2 topic (adjust if needed)
    rospy.Subscriber('/velodyne_points', PointCloud2, pointcloud_callback, tf_listener)

    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    main()
