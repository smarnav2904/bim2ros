import multiprocessing
import ifcopenshell
import ifcopenshell.geom
import numpy as np
from collections import defaultdict
import json

# Constants
RES = 0.05
GRID_SIZEX = 30
GRID_SIZEY = 30
GRID_SIZEZ = 30

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
    
    index = np.floor(x) + np.floor(y*grid_stepy) + np.floor(z*grid_stepz) 

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

                search_point = (float(x*RES), float(y*RES), float(z*RES))
                elements = tree.select(search_point, extend=RES/2)

                formatted_search_point = (
                        f'{search_point[0]:.1f}', 
                        f'{search_point[1]:.1f}', 
                        f'{search_point[2]:.1f}'
                    )
                if elements:
                    count +=1
                    for item in elements:
                        result_dict[item.GlobalId] += 1

                        if item.GlobalId not in global_id_to_int:
                            global_id_to_int[item.GlobalId] = current_int
                            current_int += 1

                        assigned_int = global_id_to_int[item.GlobalId]

                        index = point2grid(x, y, z, onedivres, grid_stepy, grid_stepz)
    
                        semantic_grid_ints[index] = assigned_int
                        print(f'{formatted_search_point} -> {assigned_int} -> {index}')

                
    bien = 0
    mal = 0            
    for i in semantic_grid_ints:
        
        if i:
            bien +=1
        else:
            mal += 1

    print(bien)
    print(mal)

    return result_dict, global_id_to_int, semantic_grid_ints, semantic_grid_zeros

# Saving Results
def save_results(global_id_to_int, result_dict, semantic_grid_ints, semantic_grid_zeros):
    with open('global_id_mapping.json', 'w') as file:
        json.dump(global_id_to_int, file, indent=4)

    with open('result_dict.json', 'w') as file:
        json.dump(result_dict, file, indent=4)

    # Save the semantic grids as npy files
    np.save('semantic_grid_ints.npy', semantic_grid_ints)
    np.save('semantic_grid_zeros.npy', semantic_grid_zeros)

# Main Execution
if __name__ == "__main__":
    ifc_file_path = 'models/casoplon.ifc'
    ifc_file, settings = setup_ifc_geometry(ifc_file_path)
    tree = initialize_iterator(settings, ifc_file)

    onedivres = 1 / RES
    grid_stepy = GRID_SIZEX * onedivres
    grid_stepz = (GRID_SIZEX*onedivres) * (GRID_SIZEY*onedivres)

    result_dict, global_id_to_int, semantic_grid_ints, semantic_grid_zeros = process_points(tree, onedivres, grid_stepy, grid_stepz)
    save_results(global_id_to_int, result_dict, semantic_grid_ints, semantic_grid_zeros)
