import matplotlib
matplotlib.use('TkAgg')

from src.geometry.curves import Curve

import numpy as np
import matplotlib.pyplot as plt


def plot_curve(curve: Curve, interval: np.ndarray, title: str = '',
               save_path: str = None, fig: plt.Figure = None,
               ax: plt.Axes = None, show: bool = True) -> None:
    """

    :param curve:
    :param interval:
    :param title:
    :param save_path:
    :param fig:
    :param ax:
    :param show:

    :return
    """

    # Get the curve to be plotted
    x, y = curve.generate_curve(interval)

    if fig is None:
        fig, ax = plt.subplots(figsize=[11, 11])

    ax.plot(x, y)
    ax.set_xlabel("X[t]")
    ax.set_ylabel("Y[t]")
    ax.set_title(title)

    if show:
        plt.show()

    if save_path is not None:
        if not save_path.endswith('.png'):
            save_path = save_path + '.png'

        fig.savefig(save_path, dpi=300, orientation='landscape', format='png')


def sweep_curve(curve_obj: Curve, interval: np.ndarray, params: [dict, ...] = (),
                title: str = '', save_path: str = None) -> None:
    """

    :param curve_obj:
    :param interval:
    :param params:
    :param title:
    :param save_path:

    :return
    """

    fig, ax = plt.subplots(figsize=[11, 11])

    for p in params:
        curve = curve_obj(**p)
        curve.plot_curve(interval=interval, title=title, fig=fig, ax=ax, show=False)

    plt.legend([', '.join([f"{key} = {p[key]}" for key in p]) for p in params])

    if save_path is not None:
        if not save_path.endswith('.png'):
            save_path = save_path + '.png'

        fig.savefig(save_path, dpi=300, orientation='landscape', format='png')

    plt.show()


def plot_geometric_flow(curve_obj: Curve, interval: np.ndarray,
                        title: str = '', save_path: str = None):
    """

    :param curve_obj:
    :param interval:
    :param title:
    :param save_path:
    :return:
    """

    # Get the flow & arclengths to be plotted
    x_axis = curve_obj.x_parametrization(interval)
    # s, cs = curve_obj.parametrize_by_arclength(interval)
    curvature_flow = curve_obj.curvature_t(interval)
    arc_lengths = np.array(
        [curve_obj.arc_length(interval[i:]) for i in range(2, len(interval))]
    )
    evolution_curve = curve_obj.generate_evolution_curves(interval)

    # Plot
    fig, axes = plt.subplots(nrows=3, figsize=[11, 11])

    # Plot evolution curve
    # axes[0].scatter(x_axis[2:], evolution_curve, cmap='hot_r', c=interval[2:])
    axes[0].scatter(evolution_curve[:, 0], evolution_curve[:, 1],
                    cmap='hot_r', c=interval[2:])
    # axes[0].plot(evolution_curve[:, 0], evolution_curve[:, 1])
    axes[0].set_ylabel("Y (t)")

    # Plot flow
    axes[1].scatter(interval[2:], curvature_flow, cmap='hot_r', c=interval[2:])
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Curvature(t)")

    # Plot arc-lengths
    scatter = axes[2].scatter(interval[2:], arc_lengths, cmap='hot_r', c=interval[2:])
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Arc Length (t)")

    cb = fig.colorbar(scatter, ax=axes)
    cb.set_label('Time', labelpad=-40, y=1.05, rotation=0)
    fig.suptitle(title)
    plt.show()

    if save_path is not None:
        if not save_path.endswith('.png'):
            save_path = save_path + '.png'

        fig.savefig(save_path, dpi=300, orientation='landscape', format='png')


def plot_curves_normals_vs_tangent(curve_obj: Curve, interval: np.ndarray,
                                   title: str = '', save_path: str = None):
    """

    :param curve_obj:
    :param interval:
    :param title:
    :param save_path:
    :return:
    """

    fig, axes = plt.subplots(ncols=2, figsize=[11, 11])

    x = curve_obj.x_parametrization(interval)
    y = curve_obj.y_parametrization(interval)
    tangent = curve_obj.unit_tangent_t(interval)
    normal = curve_obj.unit_normal_t(interval)

    for i in range(len(tangent)):
        axes[0].plot(x, y, 'k')
        axes[0].arrow(x=x[i + 1], y=y[i + 1],
                      dx=(x[i + 1] + tangent[i, 0]), dy=(y[i + 1] + tangent[i, 1]),
                      color='r')
        axes[0].set_title("Tangent")

    for i in range(len(normal)):
        axes[1].plot(x, y, 'k')
        axes[1].arrow(x=x[i + 1], y=y[i + 1],
                      dx=(x[i + 1] + normal[i, 0]), dy=(y[i + 1] + normal[i, 1]),
                      color='b')
        axes[1].set_title("Normal")

    fig.suptitle(title)

    if save_path is not None:
        if not save_path.endswith('.png'):
            save_path = save_path + '.png'

        fig.savefig(save_path, dpi=300, orientation='landscape', format='png')

    plt.show()
