#!/usr/bin/env python3

import numpy as np
import rospy
import rospkg
import networkx as nx
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def load_data(file_path):
    """Load the .npy file containing graph data."""
    return np.load(file_path)

def build_cost_matrix(data, scale_factor=10):
    """Create a cost matrix from graph data."""
    # Scale points down by dividing by the scale factor
    scaled_data = data.copy()
    scaled_data[:, :6] /= scale_factor  # Scale x1, y1, z1, x2, y2, z2

    points = np.unique(scaled_data[:, :6].reshape(-1, 3), axis=0)
    point_index = {tuple(p): i for i, p in enumerate(points)}
    num_points = len(points)
    
    # Initialize cost matrix with -1
    cost_matrix = np.full((num_points, num_points), -1.0)
    
    for row in scaled_data:
        x1, y1, z1, x2, y2, z2, cost = row
        idx1, idx2 = point_index[(x1, y1, z1)], point_index[(x2, y2, z2)]
        cost_matrix[idx1, idx2] = cost
        cost_matrix[idx2, idx1] = cost  # Assuming undirected graph
    
    return cost_matrix, points

def create_mst(cost_matrix):
    """Generate the Minimum Spanning Tree using NetworkX."""
    G = nx.Graph()
    for i in range(len(cost_matrix)):
        for j in range(len(cost_matrix)):
            if cost_matrix[i][j] != -1:
                G.add_edge(i, j, weight=cost_matrix[i][j])
    mst = nx.minimum_spanning_tree(G)
    return mst

def save_mst_to_file(mst, points, file_path):
    """Save MST edges and their costs to a text file."""
    with open(file_path, "w") as f:
        for edge in mst.edges(data=True):
            idx1, idx2, data = edge
            p1, p2 = points[idx1], points[idx2]
            cost = data["weight"]
            f.write(f"{p1.tolist()} -> {p2.tolist()} : {cost}\n")

def publish_mst(points, mst, publisher):
    """Publish the Minimum Spanning Tree as a LINE_STRIP Marker."""
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "mst"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.1
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0

    for edge in mst.edges:
        idx1, idx2 = edge
        p1, p2 = points[idx1], points[idx2]
        point1 = Point(x=p1[0], y=p1[1], z=p1[2])
        point2 = Point(x=p2[0], y=p2[1], z=p2[2])
        marker.points.extend([point1, point2])

    publisher.publish(marker)

def main():
    rospy.init_node("bim2ros_mst_visualizer")
    pub = rospy.Publisher("mst_marker", Marker, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    # Use rospkg to find the package path
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('bim2ros')  # Replace 'BIM2ROS' with your actual package name
    file_path = f"{package_path}/grids/global_graph.npy"
    mst_file_path = f"{package_path}/grids/mst_edges.txt"
    
    try:
        data = load_data(file_path)
    except FileNotFoundError:
        rospy.logerr(f"File not found: {file_path}")
        return
    
    cost_matrix, points = build_cost_matrix(data)
    mst = create_mst(cost_matrix)

    # Save MST edges to a file
    save_mst_to_file(mst, points, mst_file_path)
    rospy.loginfo(f"MST edges saved to: {mst_file_path}")
    
    rospy.loginfo("Publishing MST visualization...")
    while not rospy.is_shutdown():
        publish_mst(points, mst, pub)
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
