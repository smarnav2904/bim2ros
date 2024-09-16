#!/usr/bin/python3
import numpy as np
import ifcopenshell
import ifcopenshell.util.shape
import rospy
import random
import json
import os


def setup_ifc_geometry(ifc_file_path):
    try:
        ifc_file = ifcopenshell.open(ifc_file_path)
    except Exception as e:
        raise FileNotFoundError(f"Error opening IFC file: {ifc_file_path}. Exception: {e}")
    
    settings = ifcopenshell.geom.settings()
    settings.set(settings.DISABLE_OPENING_SUBTRACTIONS, False)
    settings.set(settings.USE_WORLD_COORDS, True)
    return ifc_file, settings


def initialize_iterator(settings, ifc_file):
    iterator = ifcopenshell.geom.iterator(settings, ifc_file)
    tree = ifcopenshell.geom.tree()
    if iterator.initialize():
        while True:
            shape = iterator.get_native()
            tree.add_element(shape)
            if not iterator.next():
                break
    return tree


def load_ifc_model(model_path):
    try:
        model = ifcopenshell.open(model_path)
        rospy.loginfo("IFC model loaded.")
        return model
    except Exception as e:
        rospy.logerr(f"Failed to load IFC model: {model_path}. Error: {e}")
        raise


def obtener_claves_por_valor(ruta_archivo, valor_buscar):
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"JSON file not found: {ruta_archivo}")

    with open(ruta_archivo, 'r') as archivo:
        data = json.load(archivo)

    claves_encontradas = [clave for clave, valor in data.items() if valor == valor_buscar]
    return claves_encontradas



def work(points, grid_data, ifc_tree, onedivres, grid_stepy, grid_stepz):
    count = 0
    for point in points:
        
        index = point2grid(point[0],point[1],point[2], onedivres, grid_stepy, grid_stepz)
        val1 = grid_data[index]

        if val1 != 0:
            val1 = obtener_claves_por_valor("global_id_mapping.json", val1)
            val1 = str(val1[0]) if val1 else "0"
        
        val2 = ifc_tree.select(point, extend=0.05)
        val2 = val2[0].GlobalId if val2 else "0"
        
        print(f'Discret: {val1} vs Continuous: {val2}')
        
        if val1 != val2:
            count += 1

    print(f"Mismatch count: {count}")
    return []


def point2grid(x, y, z, onedivres, grid_stepy, grid_stepz):
    
    index = np.floor(x * onedivres) + np.floor(y*onedivres*grid_stepy) + np.floor(z*onedivres*grid_stepz) 

    return int(index)


def read_points_from_file(file_path):
    """Reads points from a file in the format: x y z"""
    points = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                x, y, z = map(float, line.strip().split())
                points.append((x, y, z))
    except FileNotFoundError:
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading points from file {file_path}: {e}")
    return points


def main():
    # Constants
    RES = 0.05
    GRID_SIZEX = 30
    GRID_SIZEY = 30
    GRID_SIZEZ = 30

    onedivres = 1 / RES
    grid_stepy = GRID_SIZEX * onedivres
    grid_stepz = (GRID_SIZEX*onedivres) * (GRID_SIZEY*onedivres)

    # Load grid data
    grid_data_path = "semantic_grid_ints.npy"
    try:
        grid_data = np.load(grid_data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Grid data file not found: {grid_data_path}")

    # Load IFC model
    ifc_file_path = "models/casoplon.ifc"
    ifc_file, settings = setup_ifc_geometry(ifc_file_path)
    tree = initialize_iterator(settings, ifc_file)

    # Read points from file
    points_file_path = "velodyne_points_xyz_map_frame.txt"
    points = read_points_from_file(points_file_path)

    # Process points
    work(points, grid_data, tree, onedivres, grid_stepy, grid_stepz)


if __name__ == "__main__":
    main()
