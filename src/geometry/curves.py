from abc import ABC
from typing import Callable
from scipy.integrate import quad
from autograd import elementwise_grad as egrad

import autograd.numpy as np


class Curve(ABC):
    """
    A class describing a curve via explicitly describing its x & y parameterizations
    """

    def __init__(self, x_parametrization: Callable, y_parametrization: Callable,
                 curvature_computation_method: str = 'xy', grad_max: float = 1e3,
                 eps: float = 1e-6, grad_window_size: int = 100):
        """
        Initialize a curve object

        :param x_parametrization:
        :param y_parametrization:
        :param curvature_computation_method:
        :param grad_max:
        :param grad_window_size:
        """

        assert curvature_computation_method in ('xy', 'c_prime_sq')

        self.x_parametrization = x_parametrization
        self.y_parametrization = y_parametrization
        self.curvature_computation_method = curvature_computation_method
        self.grad_max = grad_max
        self.eps = eps
        self.grad_window_size = grad_window_size

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

    def _generate_x(self, interval: np.ndarray) -> np.ndarray:
        """

        """

        x = self.x_parametrization(interval)

        return x

    def _generate_y(self, interval: np.ndarray) -> np.ndarray:
        """

        """

        y = self.y_parametrization(interval)

        return y

    def generate_curve(self, interval: np.ndarray) -> (np.ndarray, np.ndarray):
        """

        :param interval:

        :return
        """

        x = self._generate_x(interval)
        y = self._generate_y(interval)

        return x, y

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

        :param t:
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

    def _stabilize_window(self, window: np.ndarray) -> float:
        """

        :param window:
        :return:
        """

        val = np.mean(window).item()
        std = np.std(window).item()
        sign = np.sign(window[len(window) // 2]).item()

        return val + (sign * std)

    def _stabilize_gradients(self, grads: np.ndarray) -> np.ndarray:
        """
        Utility method for stabilizing

        :param grads:
        :return:
        """

        # Create a copy so as not to jeapordize the original gradients
        stable_grads = grads.copy()

        # Find indices of troublesome gradients
        high_unstable_inds = np.where(np.abs(stable_grads) > self.grad_max)[0]
        low_unstable_inds = np.where(((1 / np.abs(stable_grads)) + self.eps) >
                                     self.grad_max)[0]

        # Pad with zeros and smooth over unstable values
        zeros = np.zeros((self.grad_window_size,))
        padded_stable_grads = np.concatenate([zeros, stable_grads, zeros])
        padded_stable_grads[self.grad_window_size:-self.grad_window_size] = [
            self._stabilize_window(
                padded_stable_grads[
                 (i - self.grad_window_size // 2):(i + self.grad_window_size // 2)])
            for i in range(self.grad_window_size,
                           len(padded_stable_grads) - self.grad_window_size)
        ]
        stable_grads[high_unstable_inds] = [
            padded_stable_grads[ind + self.grad_window_size]
            for ind in high_unstable_inds]
        stable_grads[low_unstable_inds] = [
            padded_stable_grads[ind + self.grad_window_size]
            for ind in low_unstable_inds]

        # Find indices of gradients which are still troublesome
        high_unstable_inds = np.where(np.abs(stable_grads) > self.grad_max)[0]
        low_unstable_inds = np.where(((1 / np.abs(stable_grads)) + self.eps) >
                                     self.grad_max)[0]

        # Clip extreme values
        stable_grads[high_unstable_inds] = (np.sign(grads[low_unstable_inds]) *
                                            self.grad_max)
        stable_grads[low_unstable_inds] = (np.sign(grads[low_unstable_inds]) *
                                           (1 / self.grad_max))

        return stable_grads

    def _diff_xy(self, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        """

        :param x:
        :param y:
        :return:
        """

        x_diff = x[1:] - x[:-1]
        y_diff = y[1:] - y[:-1]

        return x_diff, y_diff

    def grad(self, t: np.ndarray) -> (np.ndarray, np.ndarray):
        """

        :param t:
        :return:
        """

        x_grad = egrad(self.x_parametrization)(t)
        y_grad = egrad(self.y_parametrization)(t)

        return x_grad, y_grad

    def grad_sq(self, t: np.ndarray) -> (np.ndarray, np.ndarray):
        """

        :param t:
        :return:
        """

        x_grad = egrad(egrad(self.x_parametrization))(t)
        y_grad = egrad(egrad(self.y_parametrization))(t)

        return x_grad, y_grad


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


class Sinus(Curve):
    """

    """

    def __init__(self):
        """

        :param a:
        """

        def x_param(t: np.ndarray):
            return t

        def y_param(t: np.ndarray):
            return np.sin(t)

        super(Sinus, self).__init__(x_parametrization=x_param,
                                    y_parametrization=y_param)

