import rospy
import numpy as np
from scipy.spatial import KDTree
from itertools import combinations
import os
import logging
from typing import List, Tuple, Dict, Optional, Any
import roslib
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker

PACKAGE_NAME = 'bim2ros'


def bresenham_3d(x1: int, y1: int, z1: int, x2: int, y2: int, z2: int) -> List[Tuple[int, int, int]]:
    """3D Bresenham's algorithm to calculate a line between two points."""
    points = []
    x, y, z = x1, y1, z1
    dx, dy, dz = abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)
    xs, ys, zs = (1 if x2 > x1 else -1), (1 if y2 > y1 else -1), (1 if z2 > z1 else -1)

    if dx >= dy and dx >= dz:
        p1, p2 = 2 * dy - dx, 2 * dz - dx
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
        p1, p2 = 2 * dx - dy, 2 * dz - dy
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
        p1, p2 = 2 * dy - dz, 2 * dx - dz
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


def move_along_direction_vector(c1: Tuple[int, int, int], c2: Tuple[int, int, int], step_size: float = 1.0) -> List[Tuple[int, int, int]]:
    """Move along a direction vector using Bresenham's algorithm."""
    x1, y1, z1 = c1
    x2, y2, z2 = c2
    return bresenham_3d(x1, y1, z1, x2, y2, z2)[::int(step_size)]


def load_cluster_data(file_path: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Load cluster details and connections from a file."""
    try:
        data = np.load(file_path, allow_pickle=True).item()
        cluster_details = data.get('Cluster_Details', [])
        connections = np.array(data.get('Connections', []))
        return cluster_details, connections
    except Exception as e:
        rospy.logerr(f"Failed to load cluster data: {e}")
        return [], np.array([])


def build_kdtrees(cluster_details: List[Dict[str, Any]]) -> Tuple[Dict[str, KDTree], Dict[str, np.ndarray]]:
    """Build KD-trees for each cluster."""
    kdtrees = {}
    clusters = {}
    for cluster in cluster_details:
        label = cluster.get('label')
        points = np.array(cluster.get('points', []))
        if points.size > 0:
            kdtrees[label] = KDTree(points)
            clusters[label] = points
    return kdtrees, clusters


def process_ifc_points(ifc_points: np.ndarray, kdtrees: Dict[str, KDTree]) -> List[Dict[str, Any]]:
    """Find nearest clusters for IFC points."""
    cluster_connections = []
    for ifc_point in ifc_points:
        distances = []
        for label, tree in kdtrees.items():
            dist, index = tree.query(ifc_point)
            distances.append((label, dist, tree.data[index]))
        distances.sort(key=lambda x: x[1])
        if distances:
            nearest_cluster = distances[0]
            cluster_connections.append({
                'ifc_point': ifc_point.tolist(),
                'nearest_cluster': {
                    'label': nearest_cluster[0],
                    'distance': nearest_cluster[1],
                    'nearest_point': nearest_cluster[2].tolist()
                }
            })
    return cluster_connections


def check_edf_at_steps(traversed_points: List[Tuple[int, int, int]], edf: np.ndarray, radius: int = 0) -> Tuple[List[float], bool]:
    """Check EDF values at the traversed points with optional radius."""
    offsets = np.array([(i, j, k) for i in range(-radius, radius + 1)
                        for j in range(-radius, radius + 1)
                        for k in range(-radius, radius + 1)])
    edf_values = []
    for x, y, z in traversed_points:
        new_points = offsets + np.array([x, y, z])
        valid_mask = ((0 <= new_points[:, 0]) & (new_points[:, 0] < edf.shape[0]) &
                      (0 <= new_points[:, 1]) & (new_points[:, 1] < edf.shape[1]) &
                      (0 <= new_points[:, 2]) & (new_points[:, 2] < edf.shape[2]))
        valid_points = new_points[valid_mask]
        edf_values.extend(edf[tuple(valid_points.T)])
    out_of_bounds = len(edf_values) == 0
    return edf_values, out_of_bounds


def publish_line_strip(marker_pub: rospy.Publisher, final_graph: List[np.ndarray]) -> None:
    """Publish the line strip marker for visualization."""
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "global_graph"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.05
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    for c1, c2 in zip(final_graph[::2], final_graph[1::2]):
        marker.points.append(make_point(c1, 10))
        marker.points.append(make_point(c2, 10))

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        marker_pub.publish(marker)
        rate.sleep()


def make_point(coords: Tuple[float, float, float], res: float) -> Point:
    """Convert coordinates to a geometry_msgs Point."""
    point = Point()
    point.x, point.y, point.z = coords[0] / res, coords[1] / res, coords[2] / res
    return point


def calculate_cost(edf_values: List[float]) -> float:
    """Calculate traversal cost based on EDF values."""
    total = sum(edf_values)
    return len(edf_values) / total if total > 0 else float('inf')


def save_results(data: Any, output_file: str) -> None:
    """Save data to the 'grids' folder in the ROS package."""
    try:
        package_path = roslib.packages.get_pkg_dir(PACKAGE_NAME)
        grids_folder_path = os.path.join(package_path, "grids")
        os.makedirs(grids_folder_path, exist_ok=True)
        np.save(os.path.join(grids_folder_path, output_file), data)
        logging.info(f"Data saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise

def main():
    rospy.init_node('kdtree_cluster_node', anonymous=True)

    cluster_file = rospy.get_param('~cluster_file', 'path/to/your/file.npy')
    edf = np.load(rospy.get_param('~edf_file', 'path/to/your/file.npy'))
    edf_val = rospy.get_param('~edf_val', 0.1)

    if not os.path.exists(cluster_file):
        rospy.logerr(f"Cluster data file not found: {cluster_file}")
        return

    cluster_details, ifc_points = load_cluster_data(cluster_file)
    if not cluster_details or ifc_points.size == 0:
        rospy.logerr("No valid data to process.")
        return

    kdtrees, clusters = build_kdtrees(cluster_details)
    cluster_connections = process_ifc_points(ifc_points, kdtrees)

    rospy.loginfo(f"Processed {len(cluster_connections)} IFC points.")

    graph_points = []
    for connection in cluster_connections:
        c1 = [int(x) for x in connection['ifc_point']]
        c2 = [int(x) for x in connection['nearest_cluster']['nearest_point']]
        traversed_points = move_along_direction_vector(c1, c2, step_size=1)
        edf_values, out_of_bounds = check_edf_at_steps(traversed_points, edf)
        if not out_of_bounds and all(float(value) >= edf_val for value in edf_values):
            graph_points.append(np.array(c2))
            graph_points.append(np.array(c1))

    for cluster in cluster_details:
        for representative in cluster['representative']:
            graph_points.append(representative)

    pairs = list(combinations(graph_points, 2))

    marker_pub = rospy.Publisher('global_graph', Marker, queue_size=10)

    final_graph = []
    pairs_with_costs = []
    for pair in pairs:
        c1, c2 = pair
        traversed_points = move_along_direction_vector(c1, c2, step_size=1)
        edf_values, out_of_bounds = check_edf_at_steps(traversed_points, edf)
        if not out_of_bounds and all(float(value) >= edf_val for value in edf_values):
            final_graph.append(np.array(c1))
            final_graph.append(np.array(c2))
            cost = calculate_cost(edf_values)
            print(f"{c1},{c2},{cost}")

            pair_with_cost = np.concatenate([c1, c2, np.array([cost])])
        
            # Append the concatenated array to pairs_with_costs
            pairs_with_costs.append(pair_with_cost)
    
    
    save_results(pairs_with_costs, "global_graph.npy")
    publish_line_strip(marker_pub, final_graph)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
