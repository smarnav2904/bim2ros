import numpy as np
from collections import defaultdict
import os
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import roslib

PACKAGE_NAME = 'bim2ros'

def get_package_path(package_name):
    return roslib.packages.get_pkg_dir(package_name)

def load_matrices(filename):
    try:
        data = np.load(filename, allow_pickle=True).item()
        return data['adjacency_matrix'], data['cost_matrix']
    except FileNotFoundError:
        rospy.logerr(f"File '{filename}' not found.")
        return None, None
    except Exception as e:
        rospy.logerr(f"Error loading file: {e}")
        return None, None

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
        rospy.logerr(f"File error: {e}")
    except Exception as e:
        rospy.logerr(f"Error loading data: {e}")
    return None, None, None

def dfs_traversal(adjacency_matrix, cost_matrix, start_node):
    visited = set()
    traversal_path = []

    def dfs(node):
        traversal_path.append(node)
        visited.add(node)

        for neighbor in range(adjacency_matrix.shape[0]):
            # Check both adjacency and cost matrix for valid connection
            if adjacency_matrix[node, neighbor] == 1 and cost_matrix[node, neighbor] > 0 and neighbor not in visited:
                dfs(neighbor)
                traversal_path.append(node)  # Append the node again when backtracking

    dfs(start_node)
    return traversal_path

def publish_dfs_path(dfs_path, centroids):
    path_pub = rospy.Publisher('tsp_path', Path, queue_size=10)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()
        for node in dfs_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = centroids[node, 0] * rospy.get_param('resolution', 0.2)
            pose.pose.position.y = centroids[node, 1] * rospy.get_param('resolution', 0.2)
            pose.pose.position.z = centroids[node, 2] * rospy.get_param('resolution', 0.2)
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        path_pub.publish(path_msg)
        rate.sleep()

def main():
    rospy.init_node('tsp_node', anonymous=True)
    filename = os.path.join(get_package_path(PACKAGE_NAME), 'scripts/navigation/results/adjacency_and_cost_matrices.npy')
    centroids_filename = os.path.join(get_package_path(PACKAGE_NAME), 'scripts/navigation/results/centroids_data.npy')
    adjacency_matrix, cost_matrix, all_centroids = load_data(filename, centroids_filename)
    # z_upper_bound = 4 * (1/rospy.get_param('resolution', 0.2))
    # all_centroids = all_centroids[all_centroids[:, 2] < z_upper_bound]
    if adjacency_matrix is not None and cost_matrix is not None:
        start_node = rospy.get_param('initial_node', 0)
        dfs_path = dfs_traversal(adjacency_matrix, cost_matrix, start_node)
        rospy.loginfo("DFS traversal path:")
        for node in dfs_path:
            print(f"{all_centroids[node] / 5}")
        publish_dfs_path(dfs_path, all_centroids)

if __name__ == "__main__":
    main()