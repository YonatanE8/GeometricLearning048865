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

        assert np.sum(np.abs(x - t)) == pytest.approx(0, 1e-4)
        assert np.sum(np.abs(y - sin)) == pytest.approx(0, 1e-4)

    def test_grad(self, sin_curve):
        curve, interval = sin_curve
        x_grad, y_grad = curve.grad(interval)

        cos = np.cos(interval)[1:]

        assert np.sum(np.abs(x_grad - np.ones_like(x_grad))) == pytest.approx(0, 1e-6)
        assert np.sum(np.abs(y_grad - cos)) == pytest.approx(0, 1e-6)

    def test_grad_sq(self, sin_curve):
        curve, interval = sin_curve
        x_grad, y_grad = curve.grad_sq(interval)

        sin = -np.sin(interval)[2:]

        assert np.sum(np.abs(x_grad - np.zeros_like(x_grad))) == pytest.approx(0, 1e-6)
        assert np.sum(np.abs(y_grad - sin)) == pytest.approx(0, 1e-6)

    def test_tangent(self, sin_curve):
        curve, interval = sin_curve
        tangent = curve.tangent(interval)

        # Regular curve should not have any singularities
        assert len(np.where(np.sum(np.abs(tangent), 1) == 0)[0]) == 0

        # Tangent norm should always be 1
        assert np.sum(np.linalg.norm(tangent) - np.ones_like(tangent)) == \
               pytest.approx(0, 1e-8)

    def test_arclength(self, half_circle_curve, sin_curve):
        curve, interval = half_circle_curve
        arc_len = curve.arc_length(interval)
        analytic_arc_len = np.pi

        assert (np.abs(arc_len - analytic_arc_len)) == pytest.approx(0, abs=3e-2)

        curve, interval = sin_curve
        arc_len = curve.arc_length(interval)
        approximate_sol = 7.64

        assert (np.abs(arc_len - approximate_sol)) == pytest.approx(0, abs=3e-2)






