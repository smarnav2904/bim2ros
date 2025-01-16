import os
import struct
import logging
import math
import numpy as np
from tqdm import tqdm
import rospy
import roslib

PACKAGE_NAME = 'bim2ros'


class GridM:
    def __init__(self, file_name):
        """Initialize GridM with the specified file name."""
        self.file_name = file_name
        self.m_gridSize = 0
        self.m_gridSizeX = 0
        self.m_gridSizeY = 0
        self.m_gridSizeZ = 0
        self.m_sensorDev = 0.0
        self.grid_array = None

    def load_grid(self):
        """Loads grid metadata and cells from the .gridm file into a NumPy array."""
        try:
            # Check file size
            file_size = os.path.getsize(self.file_name)
            print(f"File size: {file_size} bytes")

            with open(self.file_name, 'rb') as f:
                # Read grid metadata
                self.m_gridSize = struct.unpack('i', f.read(4))[0]
                self.m_gridSizeX = struct.unpack('i', f.read(4))[0]
                self.m_gridSizeY = struct.unpack('i', f.read(4))[0]
                self.m_gridSizeZ = struct.unpack('i', f.read(4))[0]
                self.m_sensorDev = struct.unpack('f', f.read(4))[0]

                # Display metadata for debugging
                print(f"Grid Metadata: Size={self.m_gridSize}, X={self.m_gridSizeX}, Y={self.m_gridSizeY}, "
                      f"Z={self.m_gridSizeZ}, SensorDev={self.m_sensorDev}")

                # Calculate grid cell data size
                # Each grid cell has 3 floats: dist, prob, seman
                cell_size = struct.calcsize('ffi')
                expected_data_size = self.m_gridSize * cell_size

                # Check if file contains enough data for the grid cells
                if file_size - f.tell() < expected_data_size:
                    print("Error: Not enough data in file for expected grid cells")
                    return False

                # Read grid cells
                grid_data = f.read(expected_data_size)
                grid_cells = struct.unpack(
                    f'{self.m_gridSize * 3}f', grid_data)

                # Reshape grid cells into a NumPy array with 3 columns (dist, prob, seman)
                self.grid_array = np.array(
                    grid_cells, dtype=np.float32).reshape(self.m_gridSize, 3)

            print("Grid data loaded successfully.")
            return True

        except IOError:
            print(f"Error opening file {self.file_name} for reading")
            return False

        except struct.error as e:
            print(f"Error unpacking the grid data: {e}")
            return False

    def get_metadata(self):
        """Returns grid metadata as individual variables for unpacking."""
        return self.m_gridSize, self.m_gridSizeX, self.m_gridSizeY, self.m_gridSizeZ, self.m_sensorDev

    def get_grid_array(self):
        """Returns the grid data as a NumPy array (shape: [gridSize, 3])."""
        return self.grid_array

    def print_grid_cells(self, sample_size=10):
        """Prints a sample of grid cells for verification."""
        if self.grid_array is None:
            print("Grid data is not loaded.")
            return

        print(f"Printing a sample of {sample_size} grid cells:")
        for i, (dist, prob, seman) in enumerate(self.grid_array[:sample_size]):
            print(
                f"Cell {i}: Distance = {dist}, Probability = {prob}, Semantic = {seman}")


def get_package_path(package_name):
    """Retrieve the absolute path of a ROS package."""
    try:
        return roslib.packages.get_pkg_dir(package_name)
    except Exception as e:
        logging.error(f"Failed to retrieve package path: {e}")
        raise


def load_gridm_file(package_name, filename):
    """Load the gridm file from the grids folder within the package."""
    try:
        package_path = get_package_path(package_name)
        file_path = os.path.join(package_path, "grids", filename)
        if not os.path.exists(file_path):
            # Print a friendly error message if the map is not found
            print(
                f"Error: The file '{filename}' was not found in the 'grids' folder of the package '{package_name}'.")
            raise FileNotFoundError(f"{file_path} does not exist.")
        return file_path
    except Exception as e:
        logging.error(f"Failed to load gridm file: {e}")
        raise


def point_to_index(x, y, z, grid_stepy, grid_stepz):
    """Calculate a 1D index for a 3D point in a flattened array."""
    return int(x) + int((y) * grid_stepy) + int((z) * grid_stepz)


def save_results(data, output_file):
    """Save data to a specified file in the 'grids' folder of the ROS package."""
    try:
        package_path = get_package_path(PACKAGE_NAME)
        grids_folder_path = os.path.join(package_path, "grids")
        os.makedirs(grids_folder_path, exist_ok=True)
        np.save(os.path.join(grids_folder_path, output_file), data)
        logging.info(f"Data saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise


def process_points(grid, gridSizeX, gridSizeY, gridSizeZ):

    tam_x = int(gridSizeX)
    tam_y = int(gridSizeY)
    tam_z = int(gridSizeZ)

    # Use int8 (1 byte per value)
    edf = np.ones((tam_x, tam_y, tam_z), dtype=np.float16)
    occupancy_grid = np.ones((tam_x, tam_y, tam_z), dtype=np.int8)

    # tqdm added to display progress for the z-loop
    for z in tqdm(range(tam_z), desc="Processing Z-dimension", unit="z-layer"):
        for y in range(tam_y):
            for x in range(tam_x):
                index = point_to_index(x, y, z, tam_x, tam_x*tam_y)

                if (grid[index][0]) <= math.sqrt(0.1*0.1*3):
                    occupancy_grid[x][y][z] = 0  # Mark obstacle
                    edf[x][y][z] = 0
                else:
                    edf[x][y][z] = grid[index][0]

                edf[x][y][z] = grid[index][0]

    save_results(occupancy_grid, output_file="occupancy_grid.npy")
    return edf


def calculate_derivatives_3d(matrix):
    def derivative(f1, f2): return (f2 - f1) / 2.0

    rows, cols, depth = matrix.shape
    # Initialize all cells as black (0)
    change_matrix = np.zeros((rows, cols, depth))
    print(np.min(matrix))
    print(np.max(matrix))
    print(np.mean(matrix))
    print(np.median(matrix))
    print(np.std(matrix, axis=0))
    print(np.percentile(matrix, [5, 80, 85, 90, 95]))

    thresh = 0.8
    max = 50

    # Loop through each element in the matrix
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            for k in range(1, depth - 1):
                if matrix[i][j][k] > thresh and matrix[i][j][k] < max and k < 40:

                    # Calculate horizontal derivatives (XY-plane)
                    # Right neighbor
                    dx1 = derivative(matrix[i, j, k], matrix[i, j+1, k])
                    dx2 = derivative(matrix[i, j-1, k],
                                     matrix[i, j, k])  # Left neighbor

                    # Calculate vertical derivatives (XY-plane)
                    # Bottom neighbor
                    dy1 = derivative(matrix[i, j, k], matrix[i+1, j, k])
                    dy2 = derivative(matrix[i-1, j, k],
                                     matrix[i, j, k])  # Top neighbor

                    # Calculate depth (Z-axis) derivatives (XZ-plane)
                    # Forward in Z
                    dz1 = derivative(matrix[i, j, k], matrix[i, j, k + 1])
                    # Backward in Z
                    dz2 = derivative(matrix[i, j, k - 1], matrix[i, j, k])

                    condition_count = 0
                    if (np.sign(dx1) != np.sign(dx2) and np.sign(dx2) > 0 and np.sign(dx1) < 0):
                        condition_count += 1
                    if (np.sign(dy1) != np.sign(dy2) and np.sign(dy2) > 0 and np.sign(dy1) < 0):
                        condition_count += 1
                    if (np.sign(dz1) != np.sign(dz2) and np.sign(dz2) > 0 and np.sign(dz1) < 0):
                        condition_count += 1

                    if condition_count >= 1:
                        change_matrix[i, j, k] = 1  # White (change detected) 

    return change_matrix


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        # Resolve and load the map.gridm file
        gridm_file = load_gridm_file(
            PACKAGE_NAME, rospy.get_param('map', 'map.gridm'))
        gridm = GridM(gridm_file)

        if gridm.load_grid():
            # Retrieve metadata and log details
            grid_size, grid_size_x, grid_size_y, grid_size_z, sensor_dev = gridm.get_metadata()
            logging.info(
                f"Grid Metadata: Size={grid_size}, X={grid_size_x}, Y={grid_size_y}, Z={grid_size_z}, SensorDev={sensor_dev}")

            # Load grid as a NumPy array
            grid = gridm.get_grid_array()
            logging.info(f"Grid array shape: {grid.shape}")
            input("Press Enter to continue")

            edf = process_points(grid, grid_size_x, grid_size_y, grid_size_z)
            # Process the points and calculate EDF
            
            edf_3d = np.transpose(edf, (0, 1, 2))
            save_results(edf_3d, output_file="edf.npy")

            voronoi_frontier = calculate_derivatives_3d(edf_3d)
            save_results(voronoi_frontier, output_file="voronoi_frontier.npy")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        print("Make sure the map file exists in the correct folder.")
    except Exception as e:
        logging.error(f"Execution failed: {e}")
