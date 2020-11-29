import matplotlib
matplotlib.use('TkAgg')

from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt


def plot_unit_sphere() -> None:
    """
    A method for plotting a unit sphere

    :return: None
    """

    # Generate a unit sphere
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]

    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    # Plot the sphere
    fig = plt.figure(figsize=[11, 11])
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("S(u, v) = $x^{2} + y^{2} + z^{2} = 1$")
    plt.show()

