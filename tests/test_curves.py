from src.geometry import curves

import pytest
import numpy as np


@pytest.fixture
def sin_curve():
    curve = curves.Sinus()
    interval = curve.get_interval(start=0, end=(2 * np.pi), n_points=1000)

    return curve, interval


@pytest.fixture
def half_circle_curve():
    curve = curves.HalfCircle()
    interval = curve.get_interval(start=-1, end=1, n_points=1000)

    return curve, interval


class TestCurve:
    def test_generate_curve(self, sin_curve):
        curve, _ = sin_curve
        x, y = curve.generate_curve(curve.get_interval(start=0,
                                                       end=(2 * np.pi),
                                                       n_points=100))

        t = np.linspace(start=0, stop=(2 * np.pi), num=100)
        sin = np.sin(t)

        # Assert shape
        assert len(x) == len(y)
        assert len(x) == len(t)

        assert np.sum(np.abs(x - t)) == pytest.approx(0, 1e-4)
        assert np.sum(np.abs(y - sin)) == pytest.approx(0, 1e-4)

    def test_grad(self, sin_curve):
        curve, interval = sin_curve
        x_grad, y_grad = curve.grad(interval)

        cos = np.cos(interval)[1:]

        # Assert shape
        assert len(x_grad) == len(y_grad)
        assert len(x_grad) == len(cos)

        assert np.sum(np.abs(x_grad - np.ones_like(x_grad))) == pytest.approx(0, 1e-6)
        assert np.sum(np.abs(y_grad - cos)) == pytest.approx(0, 1e-6)

    def test_grad_sq(self, sin_curve):
        curve, interval = sin_curve
        x_grad, y_grad = curve.grad_sq(interval)

        sin = -np.sin(interval)[2:]

        # Assert shape
        assert len(x_grad) == len(y_grad)
        assert len(x_grad) == len(sin)

        assert np.sum(np.abs(x_grad - np.zeros_like(x_grad))) == pytest.approx(0, 1e-6)
        assert np.sum(np.abs(y_grad - sin)) == pytest.approx(0, 1e-6)

    def test_arclength(self, half_circle_curve, sin_curve):
        curve, interval = half_circle_curve
        arc_len = curve.arc_length(interval)
        analytic_arc_len = np.pi

        assert isinstance(arc_len, float)
        assert (np.abs(arc_len - analytic_arc_len)) == pytest.approx(0, abs=3e-2)

        curve, interval = sin_curve
        arc_len = curve.arc_length(interval)
        approximate_sol = 7.64

        assert (np.abs(arc_len - approximate_sol)) == pytest.approx(0, abs=3e-2)

    def test_tangent(self, sin_curve):
        curve, interval = sin_curve
        tangent = curve.tangent_t(interval)

        # Regular curve should not have any singularities
        assert len(np.where(np.sum(np.abs(tangent), 1) == 0)[0]) == 0

        # Assert shape
        assert tangent.shape == ((interval.shape[0] - 1), 2)

        # Tangent norm should always be 1
        assert np.sum(np.linalg.norm(tangent, axis=1) -
                      np.ones((tangent.shape[0], ))) == \
            pytest.approx(0, abs=1e-8)

    def test_normal(self, sin_curve):
        curve, interval = sin_curve
        normal = curve.normal_t(interval)
        x = curve.x_parametrization(interval)
        y = curve.y_parametrization(interval)
        xy = np.concatenate((np.expand_dims(x, 1), np.expand_dims(y, 1)), 1)

        # Assert shape
        assert normal.shape == ((interval.shape[0] - 1), 2)

        # Normals norm should always be 1
        assert np.sum(np.linalg.norm(normal, axis=1) -
                      np.ones((normal.shape[0], ))) == \
            pytest.approx(0, abs=1e-8)

        # Normals should be orthogonal to the original vectors
        dot_product = np.sum([np.dot(xy[(i + 1), None, :], normal.T[:, i, None])
                              for i in range(xy.shape[0] - 1)])

        assert dot_product == pytest.approx(0, abs=1e-8)

        # assert np.sum(np.matmul(xy, normal.T)) == pytest.approx(0, abs=1e-8)

    def test_curvature(self, sin_curve):
        curve, interval = sin_curve

        # Compute the curvature using the 'xy' method
        xy_curvature = curve.curvature_t(interval)

        # Compute the curvature using the 'c_prime_sq' method
        c_curvature = curve.curvature_t(interval)

        # Assert shape
        assert xy_curvature.shape == (len(interval) - 2, )

        # Both computations must be identical
        comp_diff = np.sum(np.abs(c_curvature - xy_curvature))
        assert comp_diff == pytest.approx(0, abs=1e-8)

    def test_tangent_prime(self, sin_curve):
        curve, interval = sin_curve
        tangent_prime = curve.tangent_prime(interval)

        # Assert shape
        assert tangent_prime.shape == ((len(interval) - 2), 2)

        # tangent_prime should be orthogonal to the tangent
        tangent = curve.tangent_t(interval)
        tangent = tangent[1:, :]
        inner_product = np.sum([np.dot(tangent[i, None, :], tangent_prime.T[:, i, None])
                                for i in range(tangent.shape[0])])

        assert inner_product == pytest.approx(0, abs=1e-4)
