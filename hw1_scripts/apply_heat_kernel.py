from src import PROJECT_ROOT
from src.visualizations.plot_images import plot_euler_iterations, plot_heat_kernel
import os


# Setup
image_path = os.path.join(PROJECT_ROOT, 'data', 'images', 'cameraman.jpg')
save_path = os.path.join(PROJECT_ROOT, 'data', 'images', 'cameraman_heat_kernel.gif')
t = 0.004
dt = 0.0001
dx = 1
dy = 1

# Euler iterations
# plot_euler_iterations(image_path=image_path, t=t, dt=dt, dx=dx, dy=dy,
#                       save_path=save_path)

# Heat Kernel
plot_heat_kernel(image_path=image_path, t=t, dt=dt, save_path=save_path)


