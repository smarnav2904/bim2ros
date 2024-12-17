#!/usr/bin/python3

import numpy as np
import time
import os
import roslib
import rospy

PACKAGE_NAME = 'bim2ros'

def get_package_path(package_name):
    return roslib.packages.get_pkg_dir(package_name)

def build_spanning_tree(cost_matrix):
    num_cities = cost_matrix.shape[0]
    in_mst = [False] * num_cities
    in_mst[0] = True
    mst_edges = []
    edge_weights = [float('inf')] * num_cities
    parents = [-1] * num_cities

    for i in range(1, num_cities):
        if cost_matrix[0][i] > 0:
            edge_weights[i] = cost_matrix[0][i]
            parents[i] = 0

    for _ in range(num_cities - 1):
        min_weight = float('inf')
        u = -1
        for i in range(num_cities):
            if not in_mst[i] and edge_weights[i] < min_weight and edge_weights[i] > 0:
                min_weight = edge_weights[i]
                u = i

        if u == -1:
            print("Warning: MST could not be fully constructed.")
            break

        in_mst[u] = True
        mst_edges.append((parents[u], u))

        for v in range(num_cities):
            if not in_mst[v] and cost_matrix[u][v] > 0 and cost_matrix[u][v] < edge_weights[v]:
                edge_weights[v] = cost_matrix[u][v]
                parents[v] = u

    return mst_edges


def load_data(filename, centroids_filename): 
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} not found.")
        data = np.load(filename, allow_pickle=True).item()
        adjacency_matrix, cost_matrix = data['adjacency_matrix'], data['cost_matrix']

        if not os.path.exists(centroids_filename):
            raise FileNotFoundError(f"{centroids_filename} not found.")
            
        centroids_data = np.load(centroids_filename, allow_pickle=True).item()
        all_centroids = np.concatenate((centroids_data['Cluster_Centroids'], centroids_data['IFC_Centroids'])).astype(int)

        return adjacency_matrix, cost_matrix, all_centroids
    except FileNotFoundError as e:
        print(f"File error: {e}")
    except Exception as e:
        print(f"Error loading data: {e}")
    return None, None, None



def save_mst(mst_edges, filename):
    
    try:
        np.save(filename, np.array(mst_edges))
        print(f"MST saved successfully to {filename}.")
    except Exception as e:
        print(f"Error saving MST: {e}")

def main():
    rospy.init_node('MST')

    filename = os.path.join(get_package_path(PACKAGE_NAME), 'scripts/navigation/results/adjacency_and_cost_matrices.npy')
    centroids_filename = os.path.join(get_package_path(PACKAGE_NAME), 'scripts/navigation/results/centroids_data.npy')

    mst_output_filename = os.path.join(get_package_path(PACKAGE_NAME), 'scripts/navigation/results/mst_edges.npy')  # Filename to save MST

    adjacency_matrix, cost_matrix, all_centroids = load_data(filename, centroids_filename)

    z_lower_bound = 0 * (1/rospy.get_param('resolution', 0.2))  # Example lower bound (adjust as needed)
    z_upper_bound = 4 * (1/rospy.get_param('resolution', 0.2))  # Example upper bound

    # Filter centroids within the Z bounds
    all_centroids = all_centroids[(all_centroids[:, 2] > z_lower_bound) & (all_centroids[:, 2] < z_upper_bound)]

    if adjacency_matrix is not None and cost_matrix is not None and all_centroids is not None:
        start_time = time.time()
        mst_edges = build_spanning_tree(cost_matrix)
        end_time = time.time()

        print(f"MST Edges: {mst_edges}")
        print(f"Time Taken: {end_time - start_time:.2f} seconds")

        # Save the MST to a file
        save_mst(mst_edges, mst_output_filename)

        # Plot the MST using Mayavi
        
        # plot_mst_with_mayavi(mst_edges, all_centroids)

    else:
        print("Data loading failed. Exiting...")

if __name__ == "__main__":
    main()
