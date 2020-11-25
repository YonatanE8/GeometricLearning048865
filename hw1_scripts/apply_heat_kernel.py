import matplotlib
matplotlib.use('TkAgg')

from src import PROJECT_ROOT
from src.utils.images import apply_euler_steps_central_derviative, \
    apply_2d_heat_kernel, heat_kernel

import matplotlib.pyplot as plt
import os


# --- Apply Euler iterations using the estimations of the central 2nd derivative
# Load the cameraman image
image_path = os.path.join(PROJECT_ROOT, 'data', 'images', 'cameraman.jpg')
im = plt.imread(image_path)

# Apply the heat heat_kernel using explicit Euler steps
t = 1e-7
dt = 1e-8
dx = 1
dy = 1
# images = apply_euler_steps_central_derviative(u=im, t=t, dt=dt, dx=dx, dy=dy)
#
# for image in images:
#     plt.figure()
#     plt.imshow(image, cmap='gray', vmin=0, vmax=255)
#
# plt.show()


# --- Convolve the heat heat_kernel directly with the image
images = apply_2d_heat_kernel(u=im, heat_kernel=heat_kernel,
                              t=t, dt=dt)

for image in images:
    plt.figure()
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)

plt.show()

