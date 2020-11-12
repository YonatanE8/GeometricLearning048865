from HW1 import PROJECT_ROOT
from HW1.utils.mesh import Mesh

import os
import glob


data_dir = os.path.join(PROJECT_ROOT, 'HW1', 'data', 'example_off_files')
file = glob.glob(os.path.join(data_dir, '*.off'))[0]

mesh = Mesh(file)
mesh.render_wireframe()
