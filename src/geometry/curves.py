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
                 curvature_computation_method: str = 'xy'):
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

    def tangent(self, t: np.ndarray) -> np.ndarray:
        """

        :param t:
        :return:
        """

        # Compute C'
        c_prime_x, c_prime_y = self.grad(t)
        c_prime = np.concatenate(
            [np.expand_dims(c_prime_x, 1), np.expand_dims(c_prime_y, 1)], 1
        )

        # Handle singularities
        zero_inds = np.where(np.sum(np.abs(c_prime), 1) == 0)[0]
        if len(zero_inds):
            c_prime[zero_inds] = np.ones((len(zero_inds), )) * 1e-6

        # Compute the norm
        c_prime_norm = np.linalg.norm(c_prime)

        tangent = c_prime / c_prime_norm

        return tangent

    def arc_length(self, t: np.ndarray) -> float:
        """

        :param t:
        :return:
        """

        def integrand(t: np.ndarray) -> np.ndarray:
            grad_x, grad_y = self.grad(t)
            c_prime = np.sqrt((np.power(grad_x, 2) + np.power(grad_y, 2)))

            return c_prime

        arc_length = quad(func=integrand, a=t[0], b=t[1])

        return arc_length[0]

    def normal(self, t: np.ndarray) -> np.ndarray:
        """

        :param t:
        :return:
        """

        tangent = self.tangent(t)

        # For the 2D case we can always use this
        normal = np.array(
            [-tangent[:, 1], tangent[:, 0]]
        )

        return normal

    def curvature(self, t: np.ndarray) -> np.ndarray:
        """

        :param t:
        :return:
        """

        if self.curvature_computation_method == 'xy':
            grad_x, grad_y = self.grad(t)
            grad_sq_x, grad_sq_y = self.grad_sq(t)

            nominator = grad_x * grad_sq_y - grad_y * grad_sq_x
            denominator = np.power((np.power(grad_x, 2) + np.power(grad_y, 2)), 1.5)

            curveature_ = nominator / denominator

        elif self.curvature_computation_method == 'c_prime_sq':
            normal = self.normal(t)
            grad_sq_x, grad_sq_y = self.grad_sq(t)
            grads_sq = np.concatenate(
                [np.expand_dims(grad_sq_x, 1), np.expand_dims(grad_sq_y, 1)], 1
            )

            curveature_ = np.matmul(normal, grads_sq)

        return curveature_

    def tangent_prime(self, t: np.ndarray) -> np.ndarray:
        """

        :param t:
        :return:
        """

        normal = self.normal(t)
        curvature_ = self.curvature(t)

        return curvature_ * normal


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

