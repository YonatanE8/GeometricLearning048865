import matplotlib
matplotlib.use('TkAgg')

from typing import Sequence
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


def plot_geometric_flow(curve_obj: Curve, intervals: Sequence[np.ndarray],
                        title: str = '', save_path: str = None):
    """

    :param curve_obj:
    :param intervals:
    :param title:
    :param save_path:
    :return:
    """

    # Get the flow & arclengths to be plotted
    curvature_flow = np.array(
        [curve_obj.curvature(interval) for interval in intervals]
    )
    arc_lengths = np.array(
        [curve_obj.arc_length(interval) for interval in intervals]
    )

    # Plot
    fig, axes = plt.subplots(nrows=2, sharex='col', figsize=[11, 11])

    # Plot flow
    normalized_intervals = [
        ((interval[2:] - np.min(interval[2:])) / np.max(interval[2:]))
        for interval in intervals
    ]
    [axes[0].plot(interval, curvature_flow[i])
     for i, interval in enumerate(normalized_intervals)]
    axes[0].set_ylabel("Mean Curvature Flows (t)")

    # Plot arc-lengths
    axes[1].plot(arc_lengths)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Arc Length (t)")

    fig.suptitle(title)
    plt.show()

    if save_path is not None:
        if not save_path.endswith('.png'):
            save_path = save_path + '.png'

        fig.savefig(save_path, dpi=300, orientation='landscape', format='png')

