import ifcopenshell
import ifcopenshell.geom
import numpy as np

# Setup IFC geometry
def setup_ifc_geometry(ifc_file_path):
    try:
        ifc_file = ifcopenshell.open(ifc_file_path)
    except Exception as e:
        raise ValueError(f"Failed to open IFC file: {e}")
    
    settings = ifcopenshell.geom.settings()
    settings.set(settings.DISABLE_OPENING_SUBTRACTIONS, False)
    settings.set(settings.USE_WORLD_COORDS, True)
    return ifc_file, settings

# Calculate center of vertices
def calcular_centro(vertices):
    vertices = np.array(vertices).reshape(-1, 3)
    centro = np.mean(vertices, axis=0)
    return centro

# Function to get the center points of various elements
def get_centroid_data(ifc_file_path):
    model, settings = setup_ifc_geometry(ifc_file_path)
    
    doors = model.by_type('IfcDoor')
    stairs = model.by_type('IfcStairFlight')
    
    elements = (doors, stairs)
    
    centroids = []
    
    for element_type in elements:
        for element in element_type:
            shape = ifcopenshell.geom.create_shape(settings, element)
            centroid = calcular_centro(shape.geometry.verts)
            centroids.append(centroid)
    
    return np.array(centroids)

# Example usage (if needed in a different script)
# centroids = get_centroid_data('../models/casoplonv2.ifc')
# print(centroids)
