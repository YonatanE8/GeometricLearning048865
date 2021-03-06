import matplotlib

matplotlib.use('TkAgg')

from abc import ABC, abstractmethod
from typing import Callable
from scipy.integrate import quad
from matplotlib import pyplot as plt

import numpy as np


class Curve(ABC):
    """

    """

    def __init__(self, x_parametrization: Callable, y_parametrization: Callable,
                 curvature_computation_method: str = 'xy'):
        """
        Initialize a curve object

        :param x_parametrization:
        :param y_parametrization:
        """

        assert curvature_computation_method in ('xy', 'c_prime_sq')

        self.x_parametrization = x_parametrization
        self.y_parametrization = y_parametrization
        self.curvature_computation_method = curvature_computation_method

    @staticmethod
    def get_interval(start: float = 0., end: float = 1.,
                     n_points: int = 1e4) -> np.ndarray:
        """
        :param start:
        :param end:
        :param n_points: 

        :return
        """

        interval = np.linspace(start=start, stop=end, num=n_points)

        return interval

    def generate_curve(self, interval: np.ndarray) -> (np.ndarray, np.ndarray):
        """

        :param interval:

        :return
        """

        x = self.x_parametrization(interval)
        y = self.y_parametrization(interval)

        return x, y

    def plot_curve(self, interval: np.ndarray, title: str = '',
                   save_path: str = None, fig: plt.Figure = None,
                   ax: plt.Axes = None, show: bool = True) -> None:
        """

        :param interval:
        :param title:
        :param save_path:
        :param fig:
        :param ax:
        :param show:

        :return
        """

        # Get the curve to be plotted
        x, y = self.generate_curve(interval)

        if fig is None:
            fig, ax = plt.subplots(figsize=[11, 11])
            ax.plot(x, y)
            ax.xlabel("X[t]")
            ax.ylabel("Y[t]")
            ax.title(title)

        else:
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

    def unit_tangent(self, t: np.ndarray) -> np.ndarray:
        """

        :param t:
        :return:
        """

        c_prime = self.grad(t)
        tangent = c_prime / np.linalg.norm(c_prime)

        return tangent

    def arc_length(self, t: np.ndarray) -> float:
        """

        :param t:
        :return:
        """

        def integrand(t: np.ndarray) -> np.ndarray:
            c_prime = np.linalg.norm(self.grad(t))

            return c_prime

        arc_length = quad(func=integrand, a=t[0], b=t[1])

        return arc_length[0]

    def unit_normal(self, t: np.ndarray) -> np.ndarray:
        """

        :param t:
        :return:
        """

        unit_tangent = self.unit_tangent(t)

        if len(unit_tangent) == 2:
            normal = np.array(
                [-unit_tangent[1], unit_tangent[0]]
            )

        else:
            j = np.array(
                [
                    [0, -1],
                    [1, 0]
                ]
            )

            normal = np.matmul(j, unit_tangent)

        return normal

    def curvature(self, t: np.ndarray) -> np.ndarray:
        """

        :return:
        """

        if self.curvature_computation_method == 'xy':
            g = self.grad(t)
            g_sq = self.grad_sq(t)

            nominator = g[0] * g_sq[1] - g[1] * g_sq[0]
            denominator = np.power((np.power(g[0], 2) + np.power(g[1], 2)), 1.5)

            curveature_ = nominator / denominator

        elif self.curvature_computation_method == 'c_prime_sq':
            unit_normal = self.unit_normal(t)
            grad_sq = self.grad_sq(t)

            curveature_ = np.matmul(unit_normal, grad_sq)

        return curveature_

    def unit_tangent_prime(self, t: np.ndarray) -> np.ndarray:
        """

        :param t:
        :return:
        """

        normal = self.unit_normal(t)
        curvature_ = self.curvature(t)

        return curvature_ * normal

    @abstractmethod
    def grad(self, t: np.ndarray) -> np.ndarray:
        """

        :param t:
        :return:
        """

        raise NotImplemented

    @abstractmethod
    def grad_sq(self, t: np.ndarray) -> np.ndarray:
        """

        :param t:
        :return:
        """

        raise NotImplemented


class Astroid(Curve):
    """

    """

    def __init__(self, a: float = 1.):
        """

        :param a:
        """

        def x_param(t: np.ndarray):
            return a * np.power(np.cos(t), 3)

        def y_param(t: np.ndarray):
            return a * np.power(np.sin(t), 3)

        super(Astroid, self).__init__(x_parametrization=x_param,
                                      y_parametrization=y_param)

        self.a = a

    @abstractmethod
    def grad(self, t: np.ndarray) -> np.ndarray:
        """

        :param t:
        :return:
        """

        x_prime = self.a * 3 * np.power(np.cos(t), 2) * (-np.sin(t))
        y_prime = self.a * 3 * np.power(np.sin(t), 2) * np.cos(t)

        return np.array(
            [x_prime, y_prime]
        )

    @abstractmethod
    def grad_sq(self, t: np.ndarray) -> np.ndarray:
        """

        :param t:
        :return:
        """

        x_prime_prime = (self.a * 3 *
                         ((2 * np.cos(t) * np.sin(t) * np.sin(t)) +
                          (-np.cos(t) * np.power(np.cos(t), 2))))
        y_prime_prime = (self.a * 3 *
                         ((2 * np.sin(t)) * np.cos(t) * np.cos(t)) +
                         (-np.power(np.sin(t), 2) * np.cos(t) * np.sun(t)))

        return np.array(
            [x_prime_prime, y_prime_prime]
        )


class Cardioid(Curve):
    """

    """

    def __init__(self, a: float = 1.):
        """

        :param a:
        """

        def x_param(t: np.ndarray):
            return a * (2 * np.cos(t) - np.cos(2 * t))

        def y_param(t: np.ndarray):
            return a * (2 * np.sin(t) - np.sin(2 * t))

        super(Cardioid, self).__init__(x_parametrization=x_param,
                                       y_parametrization=y_param)

        self.a = a


class Conchoid(Curve):
    """

    """

    def __init__(self, a: float = 1.):
        """

        :param a:
        """

        def x_param(t: np.ndarray):
            return a + np.cos(t)

        def y_param(t: np.ndarray):
            return a * np.tan(t) + np.sin(t)

        super(Conchoid, self).__init__(x_parametrization=x_param,
                                       y_parametrization=y_param)

        self.a = a


class Epicycloid(Curve):
    """

    """

    def __init__(self, a: float = 1., b: float = 1.):
        """

        :param a:
        :param b:
        """

        def x_param(t: np.ndarray):
            return ((a + b) * np.cos(t)) - (b * np.cos(((a / b) + 1) * t))

        def y_param(t: np.ndarray):
            return ((a + b) * np.sin(t)) - (b * np.sin(((a / b) + 1) * t))

        super(Epicycloid, self).__init__(x_parametrization=x_param,
                                         y_parametrization=y_param)

        self.a = a
        self.b = b


class Epitrochoid(Curve):
    """

    """

    def __init__(self, a: float = 1., b: float = 1., c: float = 1.):
        """

        :param a:
        :param b:
        :param c:
        """

        def x_param(t: np.ndarray):
            return ((a + b) * np.cos(t)) - (c * np.cos(((a / b) + t) * t))

        def y_param(t: np.ndarray):
            return ((a + b) * np.sin(t)) - (c * np.sin(((a / b) + 1) * t))

        super(Epitrochoid, self).__init__(x_parametrization=x_param,
                                          y_parametrization=y_param)

        self.a = a
        self.b = b
        self.c = c


class DescartesFolium(Curve):
    """

    """

    def __init__(self, a: float = 1., eps: float = 1e-8):
        """

        :param a:
        """

        def x_param(t: np.ndarray):
            singularity = np.where(t == -1.)[0]
            t[singularity] = -1 + eps
            return (3 * a * t) / (1 + np.power(t, 3))

        def y_param(t: np.ndarray):
            singularity = np.where(t == -1.)[0]
            t[singularity] = -1 + eps
            return (3 * a * np.power(t, 2)) / (1 + np.power(t, 3))

        super(DescartesFolium, self).__init__(x_parametrization=x_param,
                                              y_parametrization=y_param)

        self.a = a
        self.eps = eps


class Hypocycloid(Curve):
    """

    """

    def __init__(self, a: float = 1., b: float = 1.):
        """

        :param a:
        :param b:
        """

        def x_param(t: np.ndarray):
            return ((a - b) * np.cos(t)) + (b * np.cos(((a / b) - 1) * t))

        def y_param(t: np.ndarray):
            return ((a - b) * np.sin(t)) - (b * np.sin(((a / b) - 1) * t))

        super(Hypocycloid, self).__init__(x_parametrization=x_param,
                                          y_parametrization=y_param)

        self.a = a
        self.b = b


class Hypotrochoid(Curve):
    """

    """

    def __init__(self, a: float = 1., b: float = 1., c: float = 1.):
        """

        :param a:
        :param b:
        :param c:
        """

        def x_param(t: np.ndarray):
            return ((a - b) * np.cos(t)) + (c * np.cos(((a / b) - 1) * t))

        def y_param(t: np.ndarray):
            return ((a - b) * np.sin(t)) - (c * np.sin(((a / b) - 1) * t))

        super(Hypotrochoid, self).__init__(x_parametrization=x_param,
                                           y_parametrization=y_param)

        self.a = a
        self.b = b
        self.c = c


class InvoluteCircle(Curve):
    """

    """

    def __init__(self, a: float = 1.):
        """

        :param a:
        """

        def x_param(t: np.ndarray):
            return a * (np.cos(t) + (t * np.sin(t)))

        def y_param(t: np.ndarray):
            return a * (np.sin(t) - (t * np.cos(t)))

        super(InvoluteCircle, self).__init__(x_parametrization=x_param,
                                             y_parametrization=y_param)

        self.a = a


class Cusp(Curve):
    """

    """

    def __init__(self):
        """

        """

        def x_param(t: np.ndarray):
            return np.power(t, 3)

        def y_param(t: np.ndarray):
            return np.power(t, 2)

        super(Cusp, self).__init__(x_parametrization=x_param,
                                   y_parametrization=y_param)


class Knot(Curve):
    """

    """

    def __init__(self, a: float = 1.):
        """

        :param a:
        """

        def x_param(t: np.ndarray):
            return np.power(t, 3) - a

        def y_param(t: np.ndarray):
            return np.power(t, 2) - a

        super(Knot, self).__init__(x_parametrization=x_param,
                                   y_parametrization=y_param)

        self.a = a


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

    if save_path is not None:
        if not save_path.endswith('.png'):
            save_path = save_path + '.png'

        fig.savefig(save_path, dpi=300, orientation='landscape', format='png')

    plt.show()

