from src import PROJECT_ROOT
from src.HW1.geometry.mesh import Mesh

import os
import glob


data_dir = os.path.join(PROJECT_ROOT, 'data', 'example_off_files')
file = glob.glob(os.path.join(data_dir, '*.off'))[1]

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
# mesh.render_vertices_normals(normalize=False)
# mesh.render_vertices_normals(normalize=True)

# --- Visualize faces normals
# mesh.render_faces_normals(normalize=False)
# mesh.render_faces_normals(normalize=True)

# --- Visualize Faces & Vertices areas
# mesh.render_surface(scalar_func='face_area')
# mesh.render_surface(scalar_func='vertex_area')


