from src.utils.io import read_off
from src import PROJECT_ROOT

import matplotlib
matplotlib.use('TkAgg')

import os
import glob
import numpy as np
import matplotlib.pyplot as plt


data_dir = os.path.join(PROJECT_ROOT, 'data', 'example_off_files')
file = glob.glob(os.path.join(data_dir, '*.off'))[0]
data = read_off(file)
vertices = data[0]

x_coordinates = np.array([v[0] for v in vertices])
y_coordinates = np.array([v[1] for v in vertices])
z_coordinates = np.array([v[2] for v in vertices])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_coordinates, y_coordinates, z_coordinates)
plt.show()





