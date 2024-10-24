#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import numpy as np

def publish_points_from_npy(npy_file, topic_name='centroid_points'):
    """
    Publish cluster centroids and IFC centroids to a ROS topic from a saved .npy file.
    The points are already in the 'map' frame, no transformation required.
    """
    # Initialize the ROS node
    rospy.init_node('centroid_publisher', anonymous=True)

    # Create a publisher for the PointCloud message
    pub = rospy.Publisher(topic_name, PointCloud, queue_size=10)
    
    # Load the saved data from .npy file
    try:
        data = np.load(npy_file, allow_pickle=True).item()
        cluster_centroids = data['Cluster_Centroids']
        ifc_centroids = data['IFC_Centroids']
        
        # Print the centroids for debugging purposes
        rospy.loginfo(f"Cluster Centroids: {cluster_centroids}")
        rospy.loginfo(f"IFC Centroids (scaled by 20): {ifc_centroids}")
    except Exception as e:
        rospy.logerr(f"Error loading .npy file: {e}")
        return
    
    # Create a PointCloud message and set the frame_id to 'map'
    point_cloud = PointCloud()
    point_cloud.header.frame_id = 'map'  # Points are in the 'map' frame

    # Add cluster centroids to the PointCloud message
    for centroid in cluster_centroids:
        point = Point32()
        point.x = centroid[0] / 20
        point.y = centroid[1] / 20
        point.z = centroid[2] / 20
        if point.z < 4:
            point_cloud.points.append(point)

    # Add IFC centroids to the PointCloud message
    for ifc_centroid in ifc_centroids:
        point = Point32()
        point.x = ifc_centroid[0] / 20
        point.y = ifc_centroid[1] / 20
        point.z = ifc_centroid[2] / 20
        if point.z < 4:
            point_cloud.points.append(point)

    # Specify a rate to publish (optional)
    rate = rospy.Rate(10)  # 10 Hz

    # Keep publishing the message until ROS is shutdown
    while not rospy.is_shutdown():
        point_cloud.header.stamp = rospy.Time.now()  # Update the timestamp
        pub.publish(point_cloud)
        rate.sleep()

if __name__ == "__main__":
    # Path to your .npy file
    npy_file = '/home/rva_container/rva_exchange/catkin_ws/src/bim2ros/scripts/centroids_data.npy'  # Change to the correct path if needed
    
    # Publish the centroids to the ROS topic
    publish_points_from_npy(npy_file)
