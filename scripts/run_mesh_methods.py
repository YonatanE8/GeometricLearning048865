from src import PROJECT_ROOT
from src.HW1.utils.mesh import Mesh

import os
import glob


data_dir = os.path.join(PROJECT_ROOT, 'data', 'example_off_files')
file = glob.glob(os.path.join(data_dir, '*.off'))[0]

mesh = Mesh(file)
mesh.render_wireframe()
mesh.render_pointcloud(scalar_func='degree')
mesh.render_pointcloud(scalar_func='coo')


