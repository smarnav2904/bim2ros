#!/usr/bin/env python3

import numpy as np
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import rospkg

def load_pairs_with_costs_from_ros_package(package_name, file_name):
    """
    Load pairs with costs from the specified package using rospkg.
    """
    try:
        rospack = rospkg.RosPack()
        package_path = rospack.get_path(package_name)
        file_path = f"{package_path}/grids/{file_name}"
        
        # Load the data
        data = np.load(file_path, allow_pickle=True)
        if data.ndim != 2 or data.shape[1] != 7:
            raise ValueError("Invalid data format. Expected Nx7 array.")
        
        # Parse the data into (c1, c2, cost)
        pairs_with_costs = [
            (tuple(row[:3]), tuple(row[3:6]), row[6]) for row in data
        ]
        return pairs_with_costs
    except FileNotFoundError:
        print(f"File '{file_name}' not found in package '{package_name}'.")
    except Exception as e:
        print(f"Error loading data: {e}")
    return None


def build_mst_kruskal(pairs_with_costs):
    """
    Build the Minimum Spanning Tree (MST) using Kruskal's algorithm.
    """
    if not pairs_with_costs:
        print("Error: No pairs with costs provided.")
        return [], 0

    # Sort all edges by cost
    sorted_edges = sorted(pairs_with_costs, key=lambda x: x[2])
    
    # Disjoint-set (Union-Find) data structure
    parent = {}
    rank = {}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            # Union by rank
            if rank[rootX] > rank[rootY]:
                parent[rootY] = rootX
            elif rank[rootX] < rank[rootY]:
                parent[rootX] = rootY
            else:
                parent[rootY] = rootX
                rank[rootX] += 1

    # Initialize the disjoint-set
    for c1, c2, _ in sorted_edges:
        if c1 not in parent:
            parent[c1] = c1
            rank[c1] = 0
        if c2 not in parent:
            parent[c2] = c2
            rank[c2] = 0

    mst_edges = []
    total_cost = 0

    # Process edges
    for c1, c2, cost in sorted_edges:
        if find(c1) != find(c2):
            union(c1, c2)
            mst_edges.append((c1, c2, cost))
            total_cost += cost

    return mst_edges, total_cost


def save_msts_to_txt(mst_edges, filename="mst_coordinates.txt"):
    """
    Save MST coordinates to a text file.
    """
    with open(filename, 'w') as f:
        f.write("XYZ Coordinates of the points in the MST:\n")
        for u, v, cost in mst_edges:
            f.write(f"Point {u}: ({u[0]:.2f}, {u[1]:.2f}, {u[2]:.2f})\n")
            f.write(f"Point {v}: ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})\n")
            f.write(f"Cost: {cost:.2f}\n")
    print(f"MST coordinates saved to '{filename}'.")


def make_point(coordinates, z_offset=0):
    """
    Create a Point object from 3D coordinates.
    """
    point = Point()
    point.x, point.y, point.z = coordinates[0] / 10, coordinates[1] / 10, coordinates[2] / 10
    return point


def publish_line_strip(marker_pub, mst_edges):
    """
    Publish the MST edges as a LINE_STRIP marker.
    """
    marker = Marker()
    marker.header.frame_id = "map"  # Adjust frame ID as needed
    marker.header.stamp = rospy.Time.now()
    marker.ns = "mst"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.1  # Line width
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0
    marker.color.a = 1.0

    for u, v, _ in mst_edges:
        marker.points.append(make_point(u))
        marker.points.append(make_point(v))

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        marker_pub.publish(marker)
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node('mst_visualization_node', anonymous=True)

    # Define the package and file names
    package_name = "bim2ros"
    file_name = "global_graph.npy"

    # Load pairs with costs
    pairs_with_costs = load_pairs_with_costs_from_ros_package(package_name, file_name)
    if not pairs_with_costs:
        print("Failed to load pairs with costs.")
        exit()

    # Build the MST using Kruskal's algorithm
    mst_edges, total_cost = build_mst_kruskal(pairs_with_costs)
    if not mst_edges:
        print("No MST found.")
        exit()

    # Save the MST to a file
    save_msts_to_txt(mst_edges)

    # Publish the MST as a LINE_STRIP marker
    marker_pub = rospy.Publisher('mst_marker', Marker, queue_size=10)
    publish_line_strip(marker_pub, mst_edges)
