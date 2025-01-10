import multiprocessing
import ifcopenshell
import ifcopenshell.geom
import numpy as np
from collections import defaultdict
import json
import os
import rospy
import roslib

PACKAGE_NAME = 'bim2ros'
# Constants
RES = rospy.get_param('resolution', 0.2)
GRID_SIZEX = rospy.get_param('world_sizeX', 20)
GRID_SIZEY = rospy.get_param('world_sizeY', 20)
GRID_SIZEZ = rospy.get_param('world_sizeZ', 4)

def get_package_path(package_name):
    return roslib.packages.get_pkg_dir(package_name)

# IFC File Handling
def setup_ifc_geometry(ifc_file_path):
    ifc_file = ifcopenshell.open(ifc_file_path)
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

def point2grid(x, y, z, onedivres, grid_stepy, grid_stepz):
    index = np.floor(x) + np.floor(y * grid_stepy) + np.floor(z * grid_stepz)
    return int(index)

# Grid Initialization and Processing
def process_points(tree, onedivres, grid_stepy, grid_stepz):
    result_dict = defaultdict(int)
    size = int(np.floor(GRID_SIZEX * onedivres) * np.floor(GRID_SIZEY * onedivres) * np.floor(GRID_SIZEZ * onedivres))
    
    # Create grids for storing assigned integers and zeros
    semantic_grid_ints = np.zeros(size)
    semantic_grid_zeros = np.zeros(size)
    
    count = 1
    global_id_to_int = {}
    current_int = 1

    tam_x = int(GRID_SIZEX * onedivres)
    tam_y = int(GRID_SIZEY * onedivres)
    tam_z = int(GRID_SIZEZ * onedivres)
    
    for z in range(tam_z):
        for y in range(tam_y):
            for x in range(tam_x):
                search_point = (float(x * RES), float(y * RES), float(z * RES))
                elements = tree.select(search_point, extend=RES)
                
                formatted_search_point = (
                    f'{search_point[0]:.1f}', 
                    f'{search_point[1]:.1f}', 
                    f'{search_point[2]:.1f}'
                )
                
                if elements:
                    count += 1
                    for item in elements:
                        result_dict[item.GlobalId] += 1

                        if item.GlobalId not in global_id_to_int:
                            global_id_to_int[item.GlobalId] = current_int
                            current_int += 1

                        assigned_int = global_id_to_int[item.GlobalId]
                        index = point2grid(x, y, z, onedivres, grid_stepy, grid_stepz)
                        semantic_grid_ints[index] = assigned_int
                        print(f'{formatted_search_point} -> {assigned_int} -> {index}')
                
    # Reshape semantic_grid_ints to a 3D matrix
    semantic_grid_3d = semantic_grid_ints.reshape((tam_z, tam_y, tam_x))
    
    return result_dict, global_id_to_int, semantic_grid_ints, semantic_grid_zeros, semantic_grid_3d

# Saving Results
def save_results(global_id_to_int, result_dict, semantic_grid_ints, semantic_grid_zeros, semantic_grid_3d):
    package_path = get_package_path(PACKAGE_NAME)
    grids_folder_path = os.path.join(package_path, "grids")
    os.makedirs(grids_folder_path, exist_ok=True)

    # Save JSON mappings
    with open(os.path.join(grids_folder_path, 'global_id_mapping.json'), 'w') as file:
        json.dump(global_id_to_int, file, indent=4)

    with open(os.path.join(grids_folder_path, 'result_dict.json'), 'w') as file:
        json.dump(result_dict, file, indent=4)

    # Save the semantic grids as npy files
    np.save(os.path.join(grids_folder_path, 'semantic_grid_ints.npy'), semantic_grid_ints)
    np.save(os.path.join(grids_folder_path, 'semantic_grid_zeros.npy'), semantic_grid_zeros)
    
    # Save the 3D matrix
    np.save(os.path.join(grids_folder_path, 'semantic_grid_3d.npy'), semantic_grid_3d)

# Main Execution
if __name__ == "__main__":
    rospy.init_node('sGridGeneration')
    ifc_file_path = os.path.join(get_package_path(PACKAGE_NAME), 'models/', rospy.get_param('map'))
    ifc_file, settings = setup_ifc_geometry(ifc_file_path)
    tree = initialize_iterator(settings, ifc_file)

    onedivres = 1 / RES
    grid_stepy = GRID_SIZEX * onedivres
    grid_stepz = (GRID_SIZEX * onedivres) * (GRID_SIZEY * onedivres)

    result_dict, global_id_to_int, semantic_grid_ints, semantic_grid_zeros, semantic_grid_3d = process_points(tree, onedivres, grid_stepy, grid_stepz)
    save_results(global_id_to_int, result_dict, semantic_grid_ints, semantic_grid_zeros, semantic_grid_3d)
    print("\033[92mFinished cleanly! 3D Matrix saved for visualization.\033[0m")
