import rospy
import numpy as np
from sklearn.cluster import DBSCAN
import os
from scipy.spatial.distance import cdist
import ifcopenshell
import ifcopenshell.geom
import logging
import roslib

PACKAGE_NAME = 'bim2ros'  # Replace with your actual package name


def get_package_path(package_name):
    """Retrieve the absolute path of a ROS package."""
    try:
        return roslib.packages.get_pkg_dir(package_name)
    except Exception as e:
        rospy.logerr(f"Failed to retrieve package path for '{package_name}': {e}")
        raise


def setup_ifc_geometry(ifc_file_path):
    """Set up IFC geometry settings."""
    try:
        ifc_file = ifcopenshell.open(ifc_file_path)
    except Exception as e:
        raise ValueError(f"Failed to open IFC file: {e}")

    settings = ifcopenshell.geom.settings()
    settings.set(settings.DISABLE_OPENING_SUBTRACTIONS, False)
    settings.set(settings.USE_WORLD_COORDS, True)
    return ifc_file, settings


def calcular_centro(vertices):
    """Calculate centroid and statistics of a set of vertices."""
    vertices = np.array(vertices).reshape(-1, 3)
    centro = np.mean(vertices, axis=0)
    distancias = np.linalg.norm(vertices - centro, axis=1)
    rospy.loginfo(f"Centro: {centro}")
    rospy.loginfo(f"Media de distancias: {np.mean(distancias)}")
    rospy.loginfo(f"Desviación estándar de distancias: {np.std(distancias)}")
    return centro


def get_centroid_data(ifc_file_path, thresh):
    """Retrieve centroids of elements in an IFC file."""
    model, settings = setup_ifc_geometry(ifc_file_path)
    elements = model.by_type('IfcDoor') + model.by_type('IfcWindow')

    centroids = []
    for element in elements:
        if element.OverallWidth >= thresh:
            shape = ifcopenshell.geom.create_shape(settings, element)
            centroid = calcular_centro(shape.geometry.verts)
            centroids.append(centroid)

    return np.array(centroids)


def select_capped_equidistant_points(cluster_points, max_size):
    """Select equidistant points within a cluster to cap its size."""
    num_points = int(np.ceil(len(cluster_points) / max_size))
    if len(cluster_points) <= num_points:
        return cluster_points

    sorted_indices = np.argsort(cluster_points[:, 0])
    sorted_points = cluster_points[sorted_indices]
    indices = np.linspace(0, len(sorted_points) - 1, num_points, dtype=int)
    return sorted_points[indices]


def calculate_medoids_or_capped(points, labels, min_cluster_size=10, max_size=500):
    """Calculate medoids or capped equidistant points for clusters."""
    unique_labels = set(labels)
    cluster_representatives = []

    for label in unique_labels:
        if label != -1:  # Ignore noise points labeled as -1
            cluster_points = points[labels == label]
            if len(cluster_points) >= min_cluster_size:
                if len(cluster_points) > max_size:
                    capped_points = select_capped_equidistant_points(cluster_points, max_size)
                    cluster_representatives.extend(capped_points)
                else:
                    pairwise_distances = cdist(cluster_points, cluster_points, metric='euclidean')
                    medoid_index = np.argmin(pairwise_distances.sum(axis=1))
                    cluster_representatives.append(cluster_points[medoid_index])

    return np.array(cluster_representatives)


def cluster_and_save_representatives(file_path, radius, min_samples, min_cluster_size, max_size, output_npy, ifc_file_name):
    """Cluster points and save cluster representatives."""
    package_path = get_package_path(PACKAGE_NAME)
    file_path = os.path.join(package_path, file_path)
    output_npy = os.path.join(package_path, output_npy)
    ifc_file_path = os.path.join(package_path, ifc_file_name)

    if not os.path.exists(file_path):
        rospy.logerr(f"File '{file_path}' not found.")
        return False

    try:
        change_matrix = np.load(file_path, allow_pickle=True)
    except Exception as e:
        rospy.logerr(f"Failed to load file '{file_path}': {e}")
        return False

    points = np.argwhere(change_matrix == 1)
    if len(points) == 0:
        rospy.loginfo("No points to display in 3D space.")
        return False

    clustering = DBSCAN(eps=radius, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    cluster_representatives = calculate_medoids_or_capped(
        points, labels, min_cluster_size=min_cluster_size, max_size=max_size
    )

    ifc_points = get_centroid_data(ifc_file_path, thresh=1.6)
    filtered_points = np.array([point for point in ifc_points if point[2] < 40])

    np.save(output_npy, {'Cluster_Medoids': cluster_representatives, 'Connections': filtered_points})
    rospy.loginfo(f"Cluster representatives saved to {output_npy}")
    return True


def ros_node():
    """ROS node to run clustering and representative selection."""
    rospy.init_node('cluster_representatives_node', anonymous=True)
    file_path = rospy.get_param('~input_file', 'grids/voronoi_frontier.npy')
    radius = rospy.get_param('~radius', 5)
    min_samples = rospy.get_param('~min_samples', 1)
    min_cluster_size = rospy.get_param('~min_cluster_size', 10)
    max_size = rospy.get_param('~max_size', 500)
    output_npy = rospy.get_param('~output_file', 'grids/medoids_data.npy')
    ifc_file_name = rospy.get_param('~ifc_file', 'models/casoplonv3.ifc')

    success = cluster_and_save_representatives(
        file_path, radius, min_samples, min_cluster_size, max_size, output_npy, ifc_file_name
    )
    if success:
        rospy.loginfo("Clustering and saving completed successfully.")
    else:
        rospy.logerr("Clustering and saving failed.")


if __name__ == "__main__":
    try:
        ros_node()
    except rospy.ROSInterruptException:
        pass
