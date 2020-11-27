from src.geometry.curves import Curve

import matplotlib
matplotlib.use('TkAgg')

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
