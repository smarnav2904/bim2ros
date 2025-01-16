#!/usr/bin/env python3

import numpy as np
import heapq
import rospy
import rospkg
from std_msgs.msg import String

def load_pairs_with_costs_from_ros_package(package_name, file_name):
    """
    Load the pairs with costs from the specified package using rospkg.
    """
    try:
        # Initialize rospkg to get the package path
        rospack = rospkg.RosPack()
        
        # Get the package path
        package_path = rospack.get_path(package_name)
        
        # Construct the full path to the file within the package
        file_path = f"{package_path}/grids/{file_name}"
        
        # Load the data from the .npy file
        data = np.load(file_path, allow_pickle=True)
        pairs_with_costs = []
        
        for pair in data:
            c1 = data[:3]  # Coordinates 0, 1, 2
            c2 = data[3:6]  # Coordinates 3, 4, 5
            cost = data[6]  # Cost

            pairs_with_costs.append((c1, c2, cost))
            print(f"Pair: c1 = {c1}, c2 = {c2}, Cost = {cost}")

        return pairs_with_costs
    except FileNotFoundError:
        print(f"File '{file_name}' not found in package '{package_name}'.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

def build_mst_from_pairs(pairs_with_costs):
    """
    Build the Minimum Spanning Tree (MST) using Prim's algorithm from the pairs with their costs.
    """
    # Create a graph with nodes and costs
    graph = {}
    for c1, c2, cost in pairs_with_costs:
        c1 = tuple(c1)  # Convert to tuple
        c2 = tuple(c2)  # Convert to tuple
        
        if c1 not in graph:
            graph[c1] = []
        if c2 not in graph:
            graph[c2] = []
        
        graph[c1].append((c2, cost))
        graph[c2].append((c1, cost))
    
    # Prim's algorithm initialization
    start_node = list(graph.keys())[0]  # Start with an arbitrary node
    visited = set()
    min_heap = [(float('inf'), start_node)]  # Use infinity to start
    mst_edges = []
    total_cost = 0
    
    while min_heap:
        cost, node = heapq.heappop(min_heap)
        
        if node in visited:
            continue
        
        visited.add(node)
        total_cost += cost
        
        # Add the edges to the MST
        for neighbor, edge_cost in graph[node]:
            if neighbor not in visited:
                heapq.heappush(min_heap, (edge_cost, neighbor))
                mst_edges.append((node, neighbor, edge_cost))
    
    return mst_edges, total_cost

def save_msts_to_txt(mst_edges, filename="mst_coordinates.txt"):
    """
    Save the XYZ coordinates of the points in the MST to a text file.
    """
    with open(filename, 'w') as f:
        f.write("XYZ Coordinates of the points in the MST:\n")
        for u, v, _ in mst_edges:
            f.write(f"Point {u}: ({u[0]:.2f}, {u[1]:.2f}, {u[2]:.2f})\n")
            f.write(f"Point {v}: ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})\n")
    print(f"MST coordinates saved to '{filename}'.")

def plot_mst(mst_edges):
    """
    Visualize the MST using Mayavi, represented as line strips.
    """
    from mayavi import mlab
    
    # Create a dictionary to store the points of the MST
    mst_points = {}
    
    # Create a dictionary that stores the point connections
    for u, v, cost in mst_edges:
        if u not in mst_points:
            mst_points[u] = []
        if v not in mst_points:
            mst_points[v] = []
        mst_points[u].append(v)
        mst_points[v].append(u)
    
    # Create a list of points in the order they are visited in the MST
    visited = set()
    line_points = []

    def visit(node):
        """Visit the node and add to the line_points list"""
        if node not in visited:
            visited.add(node)
            line_points.append(node)
            for neighbor in mst_points[node]:
                visit(neighbor)

    # Start from any node (let's use the first node in mst_edges)
    start_node = mst_edges[0][0]
    visit(start_node)
    
    # Extract the x, y, z coordinates of the points in the MST
    x = [point[0] for point in line_points]
    y = [point[1] for point in line_points]
    z = [point[2] for point in line_points]
    
    # Plot the MST using Mayavi
    mlab.points3d(x, y, z, mode='point', colormap='blue')
    mlab.plot3d(x, y, z, tube_radius=0.1, color=(1, 0, 0))  # Red color for edges
    mlab.show()

# ROS Integration
def publish_mst(mst_edges):
    rospy.init_node('mst_visualization_node', anonymous=True)
    pub = rospy.Publisher('mst_edges', String, queue_size=10)

    mst_str = ', '.join([f"({u}, {v}, {cost})" for u, v, cost in mst_edges])
    rospy.loginfo(f"Publishing MST: {mst_str}")
    pub.publish(mst_str)
    rospy.spin()  # Keeps the node running

if __name__ == "__main__":
    package_name = "bim2ros"  # Name of your ROS package
    file_name = "global_graph.npy"  # Name of the file inside the grids folder

    # Load the pairs with costs from the ROS package
    pairs_with_costs = load_pairs_with_costs_from_ros_package(package_name, file_name)
    
    if not pairs_with_costs:
        print("Failed to load pairs with costs.")
        exit()
    
    # Build the MST from the pairs with costs
    mst_edges, total_cost = build_mst_from_pairs(pairs_with_costs)
    
    if not mst_edges:
        print("No MST found.")
        exit()
    
    # Save the MST coordinates to a file
    save_msts_to_txt(mst_edges)
    
    # Visualize the MST
    plot_mst(mst_edges)
    
    # Publish the MST over ROS
    publish_mst(mst_edges)
