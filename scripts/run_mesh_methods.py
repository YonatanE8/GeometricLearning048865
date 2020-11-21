from src import PROJECT_ROOT
from src.HW1.geometry.mesh import Mesh

import os
import glob


data_dir = os.path.join(PROJECT_ROOT, 'data', 'example_off_files')
file = glob.glob(os.path.join(data_dir, '*.off'))[0]

# Load Mesh
mesh = Mesh(file)

# Plot Wireframe view
mesh.render_wireframe()

# --- Plot Point-cloud view
# mesh.render_pointcloud(scalar_func='degree')
# mesh.render_pointcloud(scalar_func='coo')

# --- Plot Surface view
# mesh.render_surface(scalar_func='inds')
# mesh.render_surface(scalar_func='degree')
# mesh.render_surface(scalar_func='coo')
