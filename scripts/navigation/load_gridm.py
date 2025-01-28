import os
import struct
import logging
import math
from typing import Tuple, Optional
import numpy as np
from tqdm import tqdm
import rospy
import roslib
from visualization_msgs.msg import Marker, MarkerArray  # Import ROS Marker messages

PACKAGE_NAME = 'bim2ros'

# Constants
DEFAULT_THRESHOLD: float = float(rospy.get_param('thresh', 0.1))
GRID_FOLDER = "grids"


class GridM:
    def __init__(self, file_name: str):
        """Initialize GridM with the specified file name."""
        self.file_name: str = file_name
        self.m_gridSize: int = 0
        self.m_gridSizeX: int = 0
        self.m_gridSizeY: int = 0
        self.m_gridSizeZ: int = 0
        self.m_sensorDev: float = 0.0
        self.grid_array: Optional[np.ndarray] = None

    def load_grid(self) -> bool:
        """Loads grid metadata and cells from the .gridm file into a NumPy array."""
        try:
            file_size = os.path.getsize(self.file_name)
            logging.info(f"File size: {file_size} bytes")

            with open(self.file_name, 'rb') as f:
                # Read grid metadata
                self.m_gridSize, self.m_gridSizeX, self.m_gridSizeY, self.m_gridSizeZ, self.m_sensorDev = struct.unpack(
                    'iiiii', f.read(20)
                )
                logging.info(f"Grid Metadata: Size={self.m_gridSize}, X={self.m_gridSizeX}, "
                             f"Y={self.m_gridSizeY}, Z={self.m_gridSizeZ}, SensorDev={self.m_sensorDev}")

                cell_size = struct.calcsize('ffi')
                expected_data_size = self.m_gridSize * cell_size

                if file_size - f.tell() < expected_data_size:
                    logging.error("Not enough data in file for expected grid cells.")
                    return False

                grid_data = f.read(expected_data_size)
                grid_cells = struct.unpack(f'{self.m_gridSize * 3}f', grid_data)
                self.grid_array = np.array(grid_cells, dtype=np.float32).reshape(self.m_gridSize, 3)

            logging.info("Grid data loaded successfully.")
            return True

        except IOError:
            logging.error(f"Error opening file {self.file_name} for reading.")
            return False

        except struct.error as e:
            logging.error(f"Error unpacking the grid data: {e}")
            return False

    def get_metadata(self) -> Tuple[int, int, int, int, float]:
        """Returns grid metadata."""
        return self.m_gridSize, self.m_gridSizeX, self.m_gridSizeY, self.m_gridSizeZ, self.m_sensorDev

    def get_grid_array(self) -> Optional[np.ndarray]:
        """Returns the grid data as a NumPy array."""
        return self.grid_array

    def print_grid_cells(self, sample_size: int = 10) -> None:
        """Prints a sample of grid cells."""
        if self.grid_array is None:
            logging.warning("Grid data is not loaded.")
            return

        logging.info(f"Printing a sample of {sample_size} grid cells:")
        for i, (dist, prob, seman) in enumerate(self.grid_array[:sample_size]):
            print(f"Cell {i}: Distance = {dist}, Probability = {prob}, Semantic = {seman}")


def get_package_path(package_name: str) -> str:
    """Retrieve the absolute path of a ROS package."""
    try:
        return roslib.packages.get_pkg_dir(package_name)
    except Exception as e:
        logging.error(f"Failed to retrieve package path: {e}")
        raise


def load_gridm_file(package_name: str, filename: str) -> str:
    """Load the gridm file from the grids folder within the package."""
    try:
        package_path = get_package_path(package_name)
        file_path = os.path.join(package_path, GRID_FOLDER, filename)
        if not os.path.exists(file_path):
            logging.error(f"The file '{filename}' was not found in the '{GRID_FOLDER}' folder.")
            raise FileNotFoundError(f"{file_path} does not exist.")
        return file_path
    except Exception as e:
        logging.error(f"Failed to load gridm file: {e}")
        raise


def point_to_index(x: int, y: int, z: int, grid_stepy: int, grid_stepz: int) -> int:
    """Calculate a 1D index for a 3D point in a flattened array."""
    return x + (y * grid_stepy) + (z * grid_stepz)


def save_results(data: np.ndarray, output_file: str) -> None:
    """Save data to a specified file in the 'grids' folder of the ROS package."""
    try:
        package_path = get_package_path(PACKAGE_NAME)
        grids_folder_path = os.path.join(package_path, GRID_FOLDER)
        os.makedirs(grids_folder_path, exist_ok=True)
        np.save(os.path.join(grids_folder_path, output_file), data)
        logging.info(f"Data saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise


def process_points(grid: np.ndarray, gridSizeX: int, gridSizeY: int, gridSizeZ: int) -> np.ndarray:
    """Process points in the grid and calculate an occupancy grid."""
    tam_x, tam_y, tam_z = int(gridSizeX), int(gridSizeY), int(gridSizeZ)

    edf = np.ones((tam_x, tam_y, tam_z), dtype=np.float16)
    occupancy_grid = np.ones((tam_x, tam_y, tam_z), dtype=np.int8)

    for z in tqdm(range(tam_z), desc="Processing Z-dimension", unit="z-layer"):
        for y in range(tam_y):
            for x in range(tam_x):
                index = point_to_index(x, y, z, tam_x, tam_x * tam_y)

                if (grid[index][0]) <= math.sqrt(0.1*0.1*3):
                    occupancy_grid[x][y][z] = 0  # Mark obstacle
                    edf[x][y][z] = 0
                else: 
                    edf[x][y][z] = grid[index][0]
                
                edf[x][y][z] = grid[index][0]
                
                

    save_results(occupancy_grid, output_file="occupancy_grid.npy")
    return edf


def calculate_derivatives_3d(matrix: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Calculate 3D derivatives and detect changes."""
    rows, cols, depth = matrix.shape
    change_matrix = np.zeros((rows, cols, depth), dtype=np.uint8)
    derivative = lambda f1, f2: (f2 - f1) / 2.0
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            for k in range(1, depth - 1):
                if  matrix[i, j, k] > threshold and k < 40:
                    dx1 = derivative(matrix[i, j, k], matrix[i, j+1, k])  # Right neighbor
                    dx2 = derivative(matrix[i, j-1, k], matrix[i, j, k])  # Left neighbor
                    
                    # Calculate vertical derivatives (XY-plane)
                    dy1 = derivative(matrix[i, j, k], matrix[i+1, j, k])  # Bottom neighbor
                    dy2 = derivative(matrix[i-1, j, k], matrix[i, j, k])  # Top neighbor

                    # Calculate depth (Z-axis) derivatives (XZ-plane)
                    dz1 = derivative(matrix[i, j, k], matrix[i, j, k + 1])  # Forward in Z
                    dz2 = derivative(matrix[i, j, k - 1], matrix[i, j, k])  # Backward in Z
                    
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


def publish_markers_continuously(change_matrix: np.ndarray, scale: float = 0.1, publish_rate: float = 1.0) -> None:
    """Continuously publish a MarkerArray to visualize changes in RViz."""
    pub = rospy.Publisher("voro_markers", MarkerArray, queue_size=10)
    rate = rospy.Rate(publish_rate)  # Set the publish rate in Hz
    marker_id = 0

    # Create MarkerArray only once since the data does not change
    marker_array = MarkerArray()

    for i in range(change_matrix.shape[0]):
        for j in range(change_matrix.shape[1]):
            for k in range(change_matrix.shape[2]):
                if change_matrix[i, j, k] == 1:  # Change detected
                    marker = Marker()
                    marker.header.frame_id = "map"
                    marker.header.stamp = rospy.Time.now()
                    marker.ns = "voro_markers"
                    marker.id = marker_id
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD

                    # Set the position of the marker
                    marker.pose.position.x = i * scale
                    marker.pose.position.y = j * scale
                    marker.pose.position.z = k * scale

                    # Set the scale and color of the marker
                    marker.scale.x = 0.2
                    marker.scale.y = 0.2
                    marker.scale.z = 0.2
                    marker.color.a = 1.0  # Alpha (transparency)
                    marker.color.r = 0.0  # Red
                    marker.color.g = 0.7  # Green
                    marker.color.b = 1.0  # Blue

                    marker_array.markers.append(marker)
                    marker_id += 1

    # Continuously publish the MarkerArray
    print("Publishing!")
    while not rospy.is_shutdown():
        logging.info(f"Publishing {len(marker_array.markers)} markers to RViz.")
        pub.publish(marker_array)
        rate.sleep()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rospy.init_node('path_field_gen', anonymous=True)
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
            print("Saved!")
            # Publish changes as markers to RViz continuously
            publish_markers_continuously(voronoi_frontier)

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        print("Make sure the map file exists in the correct folder.")
    except Exception as e:
        logging.error(f"Execution failed: {e}")
