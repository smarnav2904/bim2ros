import multiprocessing
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element
import numpy as np
import edt
from tqdm import tqdm  # Add tqdm for progress bar
from sklearn.cluster import DBSCAN
import os
from get_nexus import get_centroid_data  # Import the centroid function
import itertools

# Constants
RES = 0.5
GRID_SIZEX = 200
GRID_SIZEY = 60
GRID_SIZEZ = 4

def load_centroid_data(npy_file):
    try:
        data = np.load(npy_file, allow_pickle=True).item()
        return data.get('Cluster_Centroids', []), data.get('IFC_Centroids', [])
    except FileNotFoundError:
        print(f"File '{npy_file}' not found.")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
    return None, None

def get_all_pairs(centroids):
    return list(itertools.combinations(centroids, 2))

def calculate_direction_vectors(pairs):
    direction_vectors = []
    for c1, c2 in pairs:
        direction_vector = c2 - c1
        norm = np.linalg.norm(direction_vector)
        normalized_vector = direction_vector / norm if norm != 0 else direction_vector
        direction_vectors.append((c1, c2, normalized_vector))
    return direction_vectors

def bresenham_3d(x1, y1, z1, x2, y2, z2):
    points = []
    x, y, z = x1, y1, z1
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1

    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x != x2:
            points.append((x, y, z))
            x += xs
            if p1 >= 0:
                y += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y != y2:
            points.append((x, y, z))
            y += ys
            if p1 >= 0:
                x += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z != z2:
            points.append((x, y, z))
            z += zs
            if p1 >= 0:
                y += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx

    points.append((x2, y2, z2))
    return points

def move_along_direction_vector(c1, c2, step_size=1.0):
    x1, y1, z1 = c1
    x2, y2, z2 = c2
    traversed_points = bresenham_3d(x1, y1, z1, x2, y2, z2)
    return traversed_points

def check_edf_at_steps(traversed_points, edf):
    radius = 2
    edf_values, out_of_bounds = [], False
    
    for point in traversed_points:
        X, Y, Z = point
        
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                for k in range(-radius, radius + 1):
                    new_X, new_Y, new_Z = X + i, Y + j, Z + k
                    
                    if new_X < 0 or new_Y < 0 or new_Z < 0 or new_X >= edf.shape[0] or new_Y >= edf.shape[1] or new_Z >= edf.shape[2]:
                        out_of_bounds = True
                        continue
                    
                    edf_value = edf[new_X][new_Y][new_Z]
                    edf_values.append(edf_value)
    
    return edf_values, out_of_bounds

def check_adjacency_and_build_matrix(direction_vectors, edf, all_centroids, ifc_centroids, step_size=1.0, ifc_factor=1.0):
    num_centroids = len(all_centroids)
    adjacency_matrix = np.zeros((num_centroids, num_centroids), dtype=int)
    cost_matrix = np.zeros((num_centroids, num_centroids))
    centroid_map = {tuple(c): idx for idx, c in enumerate(all_centroids)}

    for c1, c2, direction_vector in direction_vectors:
        traversed_points = move_along_direction_vector(c1, c2, step_size)
        edf_values, out_of_bounds = check_edf_at_steps(traversed_points, edf)
        
        if not out_of_bounds and all(value != 0 for value in edf_values):
            idx1, idx2 = centroid_map[tuple(c1)], centroid_map[tuple(c2)]
            adjacency_matrix[idx1, idx2] = adjacency_matrix[idx2, idx1] = 1

            # Calculate distance
            distance = np.linalg.norm(np.array(c2) - np.array(c1))

            # Check if either centroid is from IFC and apply the factor
            if tuple(c1) in map(tuple, ifc_centroids) or tuple(c2) in map(tuple, ifc_centroids):
                distance *= ifc_factor

            cost_matrix[idx1, idx2] = cost_matrix[idx2, idx1] = distance

    return adjacency_matrix, cost_matrix

def save_matrices(adjacency_matrix, cost_matrix, filename):
    np.save(filename, {'adjacency_matrix': adjacency_matrix, 'cost_matrix': cost_matrix})

def calculate_centroids(points, labels):
    unique_labels = set(labels)
    centroids = []

    for label in unique_labels:
        if label != -1:  # Ignore noise points labeled as -1
            cluster_points = points[labels == label]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                centroids.append(centroid)
    
    return np.array(centroids)

def cluster_and_save_centroids(file_path, ifc_file_path, radius, min_samples=1, 
                               output_npy="centroids_data.npy", scale_factor=10):
    
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found.")
        return False
    
    try:
        change_matrix = np.load(file_path)
    except Exception as e:
        print(f"Failed to load the file: {e}")
        return False

    points = np.argwhere(change_matrix == 1)

    if len(points) == 0:
        print("No points to display in the 3D space.")
        return False

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=radius, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    # Calculate cluster centroids
    cluster_centroids = calculate_centroids(points, labels)

    try:
        # Retrieve IFC centroids
        ifc_centroids = get_centroid_data(ifc_file_path)
    except Exception as e:
        print(f"Error loading IFC centroid data: {e}")
        return False

    # Scale IFC centroids
    ifc_centroids_scaled = ifc_centroids * scale_factor

    # Save only the cluster centroids and scaled IFC centroids to .npy
    np.save(output_npy, {'Cluster_Centroids': cluster_centroids, 'IFC_Centroids': ifc_centroids_scaled})
    
    print(f"Cluster centroids and IFC centroids (scaled by {scale_factor}) saved to {output_npy}")
    
    return True

# IFC File Handling
def setup_ifc_geometry(ifc_file_path):
    try:
        ifc_file = ifcopenshell.open(ifc_file_path)
    except Exception as e:
        raise ValueError(f"Failed to open IFC file: {e}")
    
    settings = ifcopenshell.geom.settings()
    settings.set(settings.DISABLE_OPENING_SUBTRACTIONS, False)
    settings.set(settings.USE_WORLD_COORDS, True)
    return ifc_file, settings

def initialize_iterator(settings, ifc_file):
    iterator = ifcopenshell.geom.iterator(settings, ifc_file, multiprocessing.cpu_count())
    tree = ifcopenshell.geom.tree()
    
    if iterator.initialize():
        while True:
            shape = iterator.get_native()
            tree.add_element(shape)
            if not iterator.next():
                break
    return tree

# Convert 3D coordinates (x, y, z) to a 1D index in a flattened array
def point_to_index(x, y, z, grid_stepy, grid_stepz):
    return int(x) + int((y) * grid_stepy) + int((z) * grid_stepz)

# Grid Initialization and Processing
def process_points(tree, onedivres):
    tam_x = int(GRID_SIZEX * onedivres)
    tam_y = int(GRID_SIZEY * onedivres)
    tam_z = int(GRID_SIZEZ * onedivres)

    edf = np.ones((tam_x, tam_y, tam_z), dtype=np.int8)  # Use int8 (1 byte per value)
    
    # tqdm added to display progress for the z-loop
    for z in tqdm(range(tam_z), desc="Processing Z-dimension", unit="z-layer"):
        for y in range(tam_y):
            for x in range(tam_x):
                search_point = (float(x * RES), float(y * RES), float(z * RES))
                elements = tree.select(search_point)
                if elements:
                    for e in elements:
                        if e.is_a() in ['IfcWallStandardCase', 'IfcCurtainWall', 'IfcWall', 'IfcMember', 'IfcColumn', 'IfcSlab', 'IfcRoof', 'IfcRailing']:
                            edf[x][y][z] = 0  # Mark obstacle

    save_results(edf, output_file="occupancy_grid.npy")
    return edf

# Saving Results
def save_results(edf, output_file='edf.npy'):
    np.save(output_file, edf)

def update_edf(edf, onedivres):
    dt = edt.edtsq(edf, parallel=16)
    return dt

derivative = lambda f1, f2: (f2 - f1) / 2.0

def calculate_derivatives_3d(matrix):
    rows, cols, depth = matrix.shape
    change_matrix = np.zeros((rows, cols, depth))  # Initialize all cells as black (0)
    print(np.min(matrix))
    print(np.max(matrix))
    print(np.mean(matrix))
    print(np.median(matrix))
    print(np.percentile(matrix, [75, 80, 85, 90, 95]))
    thresh = np.percentile(matrix, 0)
    # Loop through each element in the matrix
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            for k in range(1, depth - 1):
                print(f'{i},{j},{k}')
                if matrix[i][j][k] > thresh:
                    # Calculate horizontal derivatives (XY-plane)
                    dx1 = derivative(matrix[i, j, k], matrix[i, j+1, k])  # Right neighbor
                    dx2 = derivative(matrix[i, j-1, k], matrix[i, j, k])  # Left neighbor
                    
                    # Calculate vertical derivatives (XY-plane)
                    dy1 = derivative(matrix[i, j, k], matrix[i+1, j, k])  # Bottom neighbor
                    dy2 = derivative(matrix[i-1, j, k], matrix[i, j, k])  # Top neighbor

                    # Calculate depth (Z-axis) derivatives (XZ-plane)
                    dz1 = derivative(matrix[i, j, k], matrix[i, j, k + 1])  # Forward in Z
                    dz2 = derivative(matrix[i, j, k - 1], matrix[i, j, k])  # Backward in Z
                    
                    # Calculate diagonal derivatives in the XY-plane
                    d_diag11 = derivative(matrix[i, j, k], matrix[i+1, j+1, k])  # Bottom-right
                    d_diag12 = derivative(matrix[i-1, j-1, k], matrix[i, j, k])  # Top-left

                    d_diag21 = derivative(matrix[i, j, k], matrix[i+1, j-1, k])  # Bottom-left
                    d_diag22 = derivative(matrix[i-1, j+1, k], matrix[i, j, k])  # Top-right

                    # Calculate diagonal derivatives in the XZ-plane
                    d_diag_xz1 = derivative(matrix[i, j, k], matrix[i+1, j, k+1])  # Bottom-right along XZ
                    d_diag_xz2 = derivative(matrix[i-1, j, k-1], matrix[i, j, k])  # Top-left along XZ

                    # Calculate diagonal derivatives in the YZ-plane
                    d_diag_yz1 = derivative(matrix[i, j, k], matrix[i, j+1, k+1])  # Right along YZ
                    d_diag_yz2 = derivative(matrix[i, j-1, k-1], matrix[i, j, k])  # Left along YZ

                    # Calculate full 3D diagonal derivatives
                    d_diag_3d1 = derivative(matrix[i, j, k], matrix[i+1, j+1, k+1])  # Moving in all X, Y, Z
                    d_diag_3d2 = derivative(matrix[i-1, j-1, k-1], matrix[i, j, k])  # Opposite direction in X, Y, Z

                    if ((np.sign(dx1) == np.sign(-dx2) and np.sign(dx1)<0) or
                        (np.sign(dy1) == np.sign(-dy2) and np.sign(dy1)<0) or
                        (np.sign(dz1) == np.sign(-dz2) and np.sign(dz1)<0)):
                        change_matrix[i, j, k] = 1  # White (change detected)


                    # Check if derivatives have the same sign
                    # if ((np.sign(dx1) == np.sign(-dx2) and np.sign(dx1)<0) or
                    #     (np.sign(dy1) == np.sign(-dy2) and np.sign(dy1)<0) or
                    #     (np.sign(dz1) == np.sign(-dz2) and np.sign(dz1)<0) or
                    #     (np.sign(d_diag11) == np.sign(-d_diag12) and np.sign(d_diag11)<0) or
                    #     (np.sign(d_diag21) == np.sign(-d_diag22) and np.sign(d_diag21)<0) or
                    #     (np.sign(d_diag_xz1) == np.sign(-d_diag_xz2) and np.sign(d_diag_xz1)<0) or
                    #     (np.sign(d_diag_yz1) == np.sign(-d_diag_yz2) and np.sign(d_diag_yz1)<0) or
                    #     (np.sign(d_diag_3d1) == np.sign(-d_diag_3d2)) and np.sign(d_diag_3d1)<0):
                    #     change_matrix[i, j, k] = 1  # White (change detected)

                

    return change_matrix

# Main Execution
if __name__ == "__main__":
    ifc_file_path = '../models/atlas_1F.ifc'
    
    # Setup IFC geometry
    ifc_file, settings = setup_ifc_geometry(ifc_file_path)
    tree = initialize_iterator(settings, ifc_file)

    # Process IFC file into grid
    onedivres = 1 / RES
    edf = process_points(tree, onedivres)

    # Update the grid to compute Euclidean distances
    edf = update_edf(edf, onedivres)
    
    save_results(edf)
    
    edf = np.transpose(edf, (0, 1, 2))

    voronoi_frontier = calculate_derivatives_3d(edf)
    save_results(voronoi_frontier, output_file="voronoi_frontier.npy")

    radius = 3
    cluster_and_save_centroids("voronoi_frontier.npy", ifc_file_path, radius, scale_factor=onedivres)

    cluster_centroids, ifc_centroids = load_centroid_data("centroids_data.npy")
    all_centroids = np.concatenate((cluster_centroids, ifc_centroids)).astype(int)
    
    # Filter centroids based on Z axis
    z_lower_bound = 0 * onedivres
    z_upper_bound = 4 * onedivres
    all_centroids = all_centroids[(all_centroids[:, 2] > z_lower_bound) & (all_centroids[:, 2] < z_upper_bound)]
    
    # Get all pairs of centroids
    centroid_pairs = get_all_pairs(all_centroids)

    # Calculate the direction vectors for each pair
    direction_vectors = calculate_direction_vectors(centroid_pairs)
    
    # Check adjacency and build the adjacency matrix and cost matrix
    adjacency_matrix, cost_matrix = check_adjacency_and_build_matrix(direction_vectors, edf, all_centroids, ifc_centroids)

    # Save the matrices to a file
    save_matrices(adjacency_matrix, cost_matrix, "adjacency_and_cost_matrices")
    
    # Print the adjacency matrix and cost matrix
    
    print(adjacency_matrix)
    
    print(cost_matrix)