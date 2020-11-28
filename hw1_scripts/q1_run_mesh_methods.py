import matplotlib
matplotlib.use('TkAgg')

from src import PROJECT_ROOT
from src.geometry.mesh import Mesh

import os
import glob


data_dir = os.path.join(PROJECT_ROOT, 'data', 'off_files')
file = glob.glob(os.path.join(data_dir, '*.off'))[0]

# Load Mesh
mesh = Mesh(file)

# --- Due to strange behaviour by PyVista, choose one visualization at a time
# Plot Wireframe view
# mesh.render_wireframe()

# --- Plot Point-cloud view
# mesh.render_pointcloud(scalar_func='degree')
# mesh.render_pointcloud(scalar_func='coo')

# --- Plot Surface view
# mesh.render_surface(scalar_func='inds')
# mesh.render_surface(scalar_func='degree')
# mesh.render_surface(scalar_func='coo')

# --- Visualize vertices normals
# mesh.render_vertices_normals(normalize=False, mag=5e7)
# mesh.render_vertices_normals(normalize=True, mag=0.01)
# mesh.render_vertices_normals(normalize=False, mag=5e7, add_norms=True)

# --- Visualize faces normals
# mesh.render_faces_normals(normalize=False, mag=5e7)
# mesh.render_faces_normals(normalize=True, mag=0.01)
# mesh.render_faces_normals(normalize=False, mag=5e7, add_norms=True)

# --- Visualize Faces & Vertices areas
# mesh.render_surface(scalar_func='face_area')
# mesh.render_surface(scalar_func='vertex_area')

# --- Visualize the Euclidean distance of every vertex from the vertices centroid
# mesh.render_distance_from_centroid()


