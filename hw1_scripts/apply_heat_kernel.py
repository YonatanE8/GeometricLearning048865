import matplotlib
matplotlib.use('TkAgg')

from src import PROJECT_ROOT
from src.utils.images import apply_explicit_2d_heat_equation

import matplotlib.pyplot as plt
import os


# Load the cameraman image
image_path = os.path.join(PROJECT_ROOT, 'data', 'images', 'cameraman.jpg')
im = plt.imread(image_path)

# Apply the heat kernel using explicit Euler steps
t = 10
dt = 0.1
dx = 1
dy = 1
images = apply_explicit_2d_heat_equation(u=im, t=t, dt=dt, dx=dx, dy=dy)

for im in images[::10]:
    plt.figure()
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)

plt.show()


