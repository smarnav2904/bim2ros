import numpy as np
import time
import os
from mayavi import mlab  # Mayavi for 3D visualization

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

def plot_mst_with_mayavi(mst_edges, centroids):
    mlab.figure(bgcolor=(1, 1, 1))
    x, y, z = centroids[:, 0], centroids[:, 1], centroids[:, 2]
    mlab.points3d(x, y, z, color=(0, 0, 1), scale_factor=1.0)

    for edge in mst_edges:
        i, j = edge
        x_vals = [centroids[i, 0], centroids[j, 0]]
        y_vals = [centroids[i, 1], centroids[j, 1]]
        z_vals = [centroids[i, 2], centroids[j, 2]]
        mlab.plot3d(x_vals, y_vals, z_vals, color=(1, 0, 0), tube_radius=0.1)

    mlab.show()

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
    """
    Saves the MST edges to a .npy file.
    Args:
        mst_edges: List of edges (u, v) representing the MST.
        filename: Path to the file where the MST will be saved.
    """
    try:
        np.save(filename, np.array(mst_edges))
        print(f"MST saved successfully to {filename}.")
    except Exception as e:
        print(f"Error saving MST: {e}")

def main():
    filename = "scripts/navigation/results/adjacency_and_cost_matrices.npy"
    centroids_filename = "scripts/navigation/results/centroids_data.npy"
    mst_output_filename = "scripts/navigation/results/mst_edges.npy"  # Filename to save MST

    adjacency_matrix, cost_matrix, all_centroids = load_data(filename, centroids_filename)

    z_lower_bound = 0 * 5  # Example lower bound (adjust as needed)
    z_upper_bound = 4 * 5  # Example upper bound

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
        plot_mst_with_mayavi(mst_edges, all_centroids)
    else:
        print("Data loading failed. Exiting...")

if __name__ == "__main__":
    main()
