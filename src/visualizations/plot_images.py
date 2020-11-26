import matplotlib
matplotlib.use('TkAgg')

from typing import Sequence
from src.utils.images import (apply_euler_steps_central_derviative,
                              apply_2d_heat_kernel,
                              heat_kernel)
from matplotlib.animation import FuncAnimation

import numpy as np
import matplotlib.pyplot as plt


def plot_images_and_surfaces(images: Sequence, surfaces: Sequence, dt: list,
                             save_path: str = None, display_acc: int = 2) -> None:
    """
    A utility method for plotting a list of images & their respective surfaces
    side-by-side. If save_path saves a GIF of the trajectory instead of displaying
     the trajectory.

    :param images: (list) A list of images to plot, where each image is given
     as a NumPy array.
    :param surfaces: (list) A list of respective surfaces to plot,
    where each surface is given as a NumPy array.
    :param dt: (list) A list of floats, specifying the temporal point for each image.
    :param save_path: (str) path to save a GIF of the computed trajectory.
    If None does not save a GIF. Defaults is None.
    :param display_acc: (int) controls how many digits to display for t. Defaults to 2.

    :return: None
    """

    if save_path is not None and not save_path.endswith('.gif'):
        save_path = save_path + '.gif'

    x = np.arange(images[0].shape[0])
    x, y = np.meshgrid(x, x)

    def init():
        """
        Initialization method for the figures

        :return: (tuple) Tuple with matplotlib.pyplot.Figure,
        matplotlib.pyplot.Axes, matplotlib.pyplot.Axes
        """

        fig = plt.figure(figsize=(11, 11))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')

        return fig, ax1, ax2

    figure, im_ax, surf_ax = init()
    if save_path is None:
        for i, image in enumerate(images):
            # Plot image
            im_ax.imshow(image, cmap='gray', vmin=0, vmax=255)
            im_ax.set_title(f"Image: t = {dt[i]:.{display_acc}f}")

            # Plot surface
            surf_ax.plot_surface(x, y, surfaces[i], cmap='gray')
            surf_ax.set_title(f"Image Surface: t = {dt[i]:.{display_acc}f}")
            surf_ax.view_init(azim=115, elev=75)

            plt.show()

            figure, im_ax, surf_ax = init()

    else:
        def animate(index: int):
            """
            update method for the animation

            :param index: (int) index of the current image

            :return: matplotlib.pyplot.Figure
            """

            im_ax.imshow(images[index], cmap='gray', vmin=0, vmax=255)
            im_ax.set_title(f"Image: t = {dt[index]:.{display_acc}f}")

            # Plot surface
            surf_ax.clear()
            surf_ax.plot_surface(x, y, surfaces[index], cmap='gray')
            surf_ax.set_title(f"Image Surface: t = {dt[index]:.{display_acc}f}")
            surf_ax.view_init(azim=115, elev=75)

        animation = FuncAnimation(figure, animate, frames=len(images))
        animation.save(save_path, dpi=150, writer='imagemagick')


def plot_euler_iterations(image_path: str, t: {float, int}=1, dt: {float, int}=0.1,
                          dx: {float, int}=1, dy: {float, int}=1,
                          save_path: str = None, display_acc: int = 2) -> None:
    """
    Plots a trajectory of the application of Euler iterations of the
     central derivative approximation to an image.

    :param image_path: (str) path to the image to be loaded
    :param t: (float/int) temporal length to traverse. Default is 1.
    :param dt: (float/int) temporal step size. Default is 0.1
    :param dx: (float/int) spatial step size column-wise. Default is 1.
    :param dy: (float/int) spatial step size row-wise. Default is 1.
    :param save_path: (str) path to save a GIF of the computed trajectory.
    If None does not save a GIF. Defaults is None.
    :param display_acc: (int) controls how many digits to display for t. Defaults to 2.

    :return: None
    """

    # Load the image
    im = plt.imread(image_path)

    # Apply Euler iterations using the estimations of the central 2nd derivative
    images = apply_euler_steps_central_derviative(u=im, t=t, dt=dt, dx=dx, dy=dy)

    # Compute the images surfaces
    surfaces = [image / np.max(image) for image in images]

    # Plot images & surfaces
    dts = [(dt * i) for i in range(len(images))]
    plot_images_and_surfaces(images=images, surfaces=surfaces, dt=dts,
                             save_path=save_path, display_acc=display_acc)


def plot_heat_kernel(image_path: str, t: {float, int}=1, dt: {float, int}=0.1,
                     save_path: str = None, display_acc: int = 4) -> None:
    """
    Plots a trajectory of the application of the 2D heat kernel to an image.
    Applies the heat kernel in the Fourier domain.

    :param image_path: (str) path to the image to be loaded
    :param t: (float/int) temporal length to traverse. Default is 1.
    :param dt: (float/int) temporal step size. Default is 0.1
    :param save_path: (str) path to save a GIF of the computed trajectory.
    If None does not save a GIF. Defaults is None.
    :param display_acc: (int) controls how many digits to display for t. Defaults to 4.

    :return: None
    """

    # Load the image
    im = plt.imread(image_path)

    # Apply Euler iterations using the estimations of the central 2nd derivative
    images = apply_2d_heat_kernel(u=im, spectral_kernel=heat_kernel, t=t, dt=dt)

    # Compute the images surfaces
    surfaces = [image / np.max(image) for image in images]

    # Plot images & surfaces
    dts = [(dt * i) for i in range(len(images))]
    plot_images_and_surfaces(images=images, surfaces=surfaces, dt=dts,
                             save_path=save_path, display_acc=display_acc)
