from abc import ABC
from typing import Callable, Sequence
from autograd import elementwise_grad as egrad

import autograd.numpy as np


class Curve(ABC):
    """
    A class describing a curve via explicitly describing its x & y parameterizations
    """

    def __init__(self, x_parametrization: Callable, y_parametrization: Callable):
        """
        Initialize a curve object

        :param x_parametrization:
        :param y_parametrization:
        """

        self.x_parametrization = x_parametrization
        self.y_parametrization = y_parametrization

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

    @staticmethod
    def _replace_infs(t: np.ndarray, signal: np.ndarray, grad: np.ndarray,
                      grad_sq: np.ndarray = None) -> np.ndarray:
        """

        :param t:
        :param signal:
        :param grad:
        :param grad_sq:
        :return:
        """

        grad_inf_inds = np.where(np.isinf(grad))[0]
        grad_sq_inf_inds = np.where(np.isinf(grad_sq))[0] if grad_sq is not None \
            else []

        # If no infs are present we can return the original signal
        if len(grad_inf_inds) == 0 and len(grad_sq_inf_inds) == 0:
            return grad[1:] if grad_sq is None else grad_sq[2:]

        # If the infs are in the first ind we can safely delete them
        if len(grad_inf_inds) and grad_inf_inds[0] == 0:
            grad_inf_inds = grad_inf_inds[1:]

        if len(grad_sq_inf_inds) and grad_sq_inf_inds[0] == 0:
            grad_sq_inf_inds = grad_sq_inf_inds[1:]

        # Correct indices
        if len(grad_inf_inds):
            grad_inf_inds -= 1

        if len(grad_sq_inf_inds):
            grad_sq_inf_inds -= 2

        # Compute dt
        dt = np.diff(t)

        # Start by fixing grads - correct every inf using empirical differentiation
        diff_signal = np.diff(signal)
        grad = grad[1:]
        grad[grad_inf_inds] = diff_signal[grad_inf_inds] / dt[grad_inf_inds]

        # Now we can fix grad_sq if relevant or just return the corrected grads
        if grad_sq is None:
            return grad

        diff_grad = np.diff(grad)
        grad_sq = grad_sq[2:]
        grad_sq[grad_sq_inf_inds] = diff_grad[grad_sq_inf_inds]

        return grad_sq

    def grad(self, t: np.ndarray) -> (np.ndarray, np.ndarray):
        """

        :param t:
        :return:
        """

        x_grad = egrad(self.x_parametrization)(t)
        y_grad = egrad(self.y_parametrization)(t)

        x_grad = self._replace_infs(t=t, signal=self.x_parametrization(t), grad=x_grad)
        y_grad = self._replace_infs(t=t, signal=self.y_parametrization(t), grad=y_grad)

        return x_grad, y_grad

    def grad_sq(self, t: np.ndarray) -> (np.ndarray, np.ndarray):
        """

        :param t:
        :return:
        """

        x_grad = egrad(egrad(self.x_parametrization))(t)
        y_grad = egrad(egrad(self.y_parametrization))(t)

        x_grad = self._replace_infs(t=t, signal=self.x_parametrization(t),
                                    grad=egrad(self.x_parametrization)(t),
                                    grad_sq=x_grad)
        y_grad = self._replace_infs(t=t, signal=self.x_parametrization(t),
                                    grad=egrad(self.y_parametrization)(t),
                                    grad_sq=y_grad)

        return x_grad, y_grad

    def grad_norm(self, t: np.ndarray) -> np.ndarray:
        """

        """

        # Compute dt
        dt = np.diff(t)

        grad_x, grad_y = self.grad(t)
        grad_x *= dt
        grad_y *= dt
        grad_norm = np.expand_dims(np.hypot(grad_x, grad_y), 1)

        return grad_norm

    def arc_length(self, t: np.ndarray) -> float:
        """

        :param t:
        :return:
        """

        # Compute ||C'(t)||
        integrand = self.grad_norm(t)

        # Compute s(t)
        arc_length = np.sum(integrand)

        return arc_length

    def parametrize_by_arclength(self, t) -> (np.ndarray, np.ndarray):
        """

        """

        # Compute s(t)
        s = np.array([
            self.arc_length(t[:i] for i in range(1, len(t)))
        ])

        # Compute C'(s(t)) == T(t) & dt
        dt = np.diff(t)
        c_prime = self.unit_tangent_t(t)

        # Compute C(s), remember that ds\dt = ||C'(t)||
        ds_dt = self.grad_norm(t)
        c_s = c_prime * ds_dt * dt
        c_s = np.cumsum(c_s)

        return s, c_s

    def unit_tangent_t(self, t: np.ndarray) -> np.ndarray:
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
        zero_inds = np.where(np.sum(np.abs(c_prime), axis=1) == 0)[0]
        if len(zero_inds):
            c_prime[zero_inds] = np.ones((len(zero_inds), )) * 1e-8

        # Compute the norm
        c_prime_norm = np.expand_dims(np.linalg.norm(c_prime, axis=1), 1)

        # Compute T(t) = C'(t) / ||C'(t)||
        tangent = c_prime / c_prime_norm

        return tangent

    def unit_normal_t(self, t: np.ndarray) -> np.ndarray:
        """

        :param t:
        :return:
        """

        # Get T(t) the unit tangent vector
        tangent = self.unit_tangent_t(t)

        # # For the 2D case we can always use this
        normal = np.concatenate([
            np.expand_dims(-tangent[:, 1], 1),
            np.expand_dims(tangent[:, 0], 1)
            ], 1)

        return normal

    def unit_tangent_s(self, t: np.ndarray) -> np.ndarray:
        """

        :param t:
        :return:
        """

        # Compute T(t)
        tangent = self.unit_tangent_t(t)

        # Compute ds/dt
        ds_dt = self.grad_norm(t)

        # T(s(t)) = ds_dt * T(t)
        tangent_s = ds_dt * tangent

        return tangent_s

    def unit_normal_s(self, t: np.ndarray) -> np.ndarray:
        """

        :param t:
        :return:
        """

        tangent = self.unit_tangent_s(t)

        # N(s) = JT(s), where J = [[0, -1], [1, 0]]
        normal = np.concatenate([
            np.expand_dims(-tangent[:, 1], 1),
            np.expand_dims(tangent[:, 0], 1)
            ], 1)

        return normal

    def curvature_t(self, t: np.ndarray) -> np.ndarray:
        """

        :param t:
        :return:
        """

        grad_x, grad_y = self.grad(t)
        grad_x, grad_y = grad_x[1:], grad_y[1:]
        grad_sq_x, grad_sq_y = self.grad_sq(t)

        nominator = grad_x * grad_sq_y - grad_y * grad_sq_x
        denominator = np.power((np.power(grad_x, 2) + np.power(grad_y, 2)), 1.5)

        curveature_ = nominator / denominator

        return curveature_

    def curvature_s(self, t: np.ndarray) -> np.ndarray:
        """

        :param t:
        :return:
        """

        # Get N(s)
        n_s = self.unit_normal_s(t)

        # Estimate C''(s), while remembering that C'(s) == T(t)
        c_prime_s = self.unit_tangent_t(t)
        c_prime_prime_s = np.diff(c_prime_s, axis=0)

        # k(s) = <N(s), C''(s)>
        curvature_s = np.concatenate([
            np.dot(n_s[i, None, :], c_prime_prime_s.T[:, i, None])
            for i in range(n_s.shape[0])], 0).squeeze()

        return curvature_s

    def tangent_prime_t(self, t: np.ndarray) -> np.ndarray:
        """

        :param t:
        :return:
        """

        normal = self.unit_normal_t(t)[1:, :]
        curvature_ = self.curvature_t(t)

        return np.expand_dims(curvature_, 1) * normal

    def generate_evolution_curves(self, t: np.ndarray) -> np.ndarray:
        """

        :param t:
        :param n_curves:
        :return:
        """

        # Compute dt
        dt = np.diff(t)[0]

        # Compute the curvature
        curvature = self.curvature_t(t)

        # Compute the evolution curve through descent iterations
        y = self.y_parametrization(t)
        evolution_curves = y[2:] - (dt * curvature)

        return evolution_curves


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
            if len(singularity):
                t[singularity] = -1 + eps

            return (3 * a * t) / (1 + np.power(t, 3))

        def y_param(t: np.ndarray):
            singularity = np.where(t == -1.)[0]

            if len(singularity):
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


class HalfCircle(Curve):
    """

    """

    def __init__(self):
        """

        """

        def x_param(t: np.ndarray):
            return t

        def y_param(t: np.ndarray):
            return np.sqrt((1 - np.power(t, 2)))

        super(HalfCircle, self).__init__(x_parametrization=x_param,
                                         y_parametrization=y_param)


class Ellipse(Curve):
    """

    """

    def __init__(self, a: float = 1., b: float = 2.):
        """

        :param a:
        :param b:
        """

        def x_param(t: np.ndarray):
            return a * np.cos(t)

        def y_param(t: np.ndarray):
            return b * np.sin(t)

        super(Ellipse, self).__init__(x_parametrization=x_param,
                                      y_parametrization=y_param)
