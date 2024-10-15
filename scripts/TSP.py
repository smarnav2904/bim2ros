import numpy as np
from collections import defaultdict
import os
import rospy
from geometry_msgs.msg import Point, PoseStamped  # Import the Point and PoseStamped message types
from nav_msgs.msg import Path  # Import the Path message type
from visualization_msgs.msg import Marker, MarkerArray  # Import Marker and MarkerArray

def load_matrices(filename):
    """
    Loads the adjacency matrix and cost matrix from a .npy file.
    Args:
        filename: Path to the .npy file containing the matrices.
    Returns:
        adjacency_matrix, cost_matrix (numpy arrays)
    """
    try:
        data = np.load(filename, allow_pickle=True).item()
        return data['adjacency_matrix'], data['cost_matrix']
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None, None

def load_mst(filename):
    """
    Loads the MST edges from a .npy file.
    Args:
        filename: Path to the .npy file containing the MST edges.
    Returns:
        A list of edges representing the MST (each edge is a tuple (i, j)).
    """
    try:
        mst_edges = np.load(filename, allow_pickle=True)
        print(f"MST loaded successfully from {filename}.")
        return mst_edges
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None
    except Exception as e:
        print(f"Error loading MST: {e}")
        return None

def find_leaves(mst_edges):
    """
    Finds the leaf nodes in the Minimum Spanning Tree (MST).
    A leaf is a node with only one connection (degree = 1).
    Args:
        mst_edges: List of edges (u, v) representing the MST.
    Returns:
        A list of nodes that are leaves.
    """
    if mst_edges is None or mst_edges.size == 0:
        return []

    num_nodes = max(max(edge) for edge in mst_edges) + 1
    degree_count = np.zeros(num_nodes, dtype=int)

    for u, v in mst_edges:
        degree_count[u] += 1
        degree_count[v] += 1

    leaves = np.where(degree_count == 1)[0]  # Find nodes with degree 1
    return leaves

def check_adjacency_between_leaves(leaves, adjacency_matrix):
    """
    Check the adjacency between the leaf nodes.
    Args:
        leaves: List of leaf node indices.
        adjacency_matrix: Full adjacency matrix of the graph.
    Returns:
        A dictionary where each leaf node maps to its adjacent leaf nodes.
    """
    leaf_adjacency = {}

    for i, leaf1 in enumerate(leaves):
        adjacent_leaves = []
        # Only check the next leaves to avoid duplicate checks
        for leaf2 in leaves[i + 1:]:
            if adjacency_matrix[leaf1, leaf2] == 1:
                adjacent_leaves.append(leaf2)
        leaf_adjacency[leaf1] = adjacent_leaves

    return leaf_adjacency

def update_mst_with_leaf_adjacency(mst_edges, leaf_adjacency):
    """
    Updates the MST by adding edges between the leaf nodes that are adjacent.
    Args:
        mst_edges: List of edges (u, v) representing the original MST.
        leaf_adjacency: Dictionary where each leaf node maps to its adjacent leaf nodes.
    Returns:
        Updated list of edges including the new leaf connections.
    """
    updated_mst_edges = mst_edges.tolist()  # Convert to list for modification

    # Add the new leaf connections
    for leaf, adjacent_leaves in leaf_adjacency.items():
        for adjacent_leaf in adjacent_leaves:
            updated_mst_edges.append((leaf, adjacent_leaf))

    return updated_mst_edges

def filter_centroids_by_elevation(centroids, z_threshold):
    return centroids[centroids[:, 2] < z_threshold]

def load_data(filename, centroids_filename):
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} not found.")
        data = np.load(filename, allow_pickle=True).item()
        adjacency_matrix, cost_matrix = data['adjacency_matrix'], data['cost_matrix']

        if not os.path.exists(centroids_filename):
            raise FileNotFoundError(f"{centroids_filename} not found.")

        centroids_data = np.load(centroids_filename, allow_pickle=True).item()
        all_centroids = np.concatenate(
            (centroids_data['Cluster_Centroids'], centroids_data['IFC_Centroids'])).astype(int)

        return adjacency_matrix, cost_matrix, all_centroids
    except FileNotFoundError as e:
        print(f"File error: {e}")
    except Exception as e:
        print(f"Error loading data: {e}")
    return None, None, None

def solve_tsp(cost_matrix, start_node):
    """
    Solves the Traveling Salesman Problem (TSP) using a nearest neighbor approach,
    ignoring edges with a cost of 0.
    Args:
        cost_matrix: The cost matrix representing distances between nodes.
        start_node: The starting node for the TSP path.
    Returns:
        A list representing the TSP path.
    """
    n = cost_matrix.shape[0]
    visited = [False] * n
    path = [start_node]
    visited[start_node] = True

    for _ in range(n - 1):
        last_node = path[-1]
        next_node = None
        min_cost = float('inf')

        for j in range(n):
            # Only consider valid edges (cost > 0) and unvisited nodes
            if not visited[j] and cost_matrix[last_node, j] > 0 and cost_matrix[last_node, j] < min_cost:
                min_cost = cost_matrix[last_node, j]
                next_node = j

        if next_node is None:
            print("No valid next node found. Exiting the TSP solving loop.")
            break  # Exit if no valid next node is found

        path.append(next_node)
        visited[next_node] = True

    return path

def publish_tsp_path(tsp_path, centroids):
    """
    Publishes the TSP path as a nav_msgs/Path to RViz using ROS.
    
    Args:
        tsp_path: List of nodes representing the TSP path.
        centroids: 2D numpy array of shape (num_nodes, 3) with the coordinates of each node.
    """
    rospy.init_node('tsp_node', anonymous=True)
    path_pub = rospy.Publisher('tsp_path', Path, queue_size=10)
    
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():  # Loop until the node is shut down
        path_msg = Path()
        path_msg.header.frame_id = "map"  # Frame of reference for RViz
        path_msg.header.stamp = rospy.Time.now()

        # Add points to the path
        for node in tsp_path:
            pose = PoseStamped()
            pose.header = path_msg.header  # Set the header for each pose
            pose.pose.position.x = centroids[node, 0] / 10
            pose.pose.position.y = centroids[node, 1] / 10
            pose.pose.position.z = centroids[node, 2] / 10
            
            # You can set orientation as needed, here it is set to zero
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            
            path_msg.poses.append(pose)

        path_pub.publish(path_msg)
        rate.sleep()  # Sleep to maintain the rate

def main():
    # Load the MST and find the leaves
    mst_filename = "/home/rva_container/rva_exchange/catkin_ws/src/bim2ros/scripts/mst_edges.npy"
    mst_edges = load_mst(mst_filename)
    leaves = find_leaves(mst_edges)

    # Load the full adjacency and cost matrices
    filename = "/home/rva_container/rva_exchange/catkin_ws/src/bim2ros/scripts/adjacency_and_cost_matrices.npy"
    centroids_filename = "/home/rva_container/rva_exchange/catkin_ws/src/bim2ros/scripts/centroids_data.npy"

    adjacency_matrix, cost_matrix, all_centroids = load_data(
        filename, centroids_filename)
    
    z_lower_bound = 0 * 10  # Example lower bound (adjust as needed)
    z_upper_bound = 4 * 10  # Example upper bound

    # Filter centroids within the Z bounds
    all_centroids = all_centroids[(all_centroids[:, 2] < z_upper_bound)]

    if adjacency_matrix is not None and leaves.size > 0:
        # Print XYZ coordinates of the leaf nodes
        print("Leaf nodes XYZ coordinates:")
        for leaf in leaves:
            print(f"Cluster XYZ: {all_centroids[leaf]}")  # Printing the XYZ coordinates

        # Check the adjacency only between the leaves in the full tree
        leaf_adjacency = check_adjacency_between_leaves(
            leaves, adjacency_matrix)

        # Update the MST with leaf adjacency
        updated_mst_edges = update_mst_with_leaf_adjacency(mst_edges, leaf_adjacency)

        # Solve the TSP
        start_node = 7  # Starting node for TSP (you can set this to any valid node)
        tsp_path = solve_tsp(cost_matrix, start_node)

        # Print the TSP path with coordinates
        print("TSP path:")
        for node in tsp_path:
            print(f"Node: {node}, Coordinates: {all_centroids[node]}")

        # Publish the TSP path to RViz
        publish_tsp_path(tsp_path, all_centroids)

if __name__ == "__main__":
    main()
