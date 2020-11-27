from src.geometry import curves

import pytest
import numpy as np


@pytest.fixture
def sin_curve():
    curve = curves.Sinus()
    interval = curve.get_interval(start=0, end=(2 * np.pi), n_points=100)

    return curve, interval


class TestCurve:
    def test_generate_curve(self, sin_curve):
        curve, _ = sin_curve
        x, y = curve.generate_curve(curve.get_interval(start=0,
                                                       end=(2 * np.pi),
                                                       n_points=100))

        t = np.linspace(start=0, stop=(2 * np.pi), num=100)
        sin = np.sin(t)

        assert pytest.approx(np.sum(np.abs(x - t)), 0, 1e-4)
        assert pytest.approx(np.sum(np.abs(y - sin)), 0, 1e-4)

    def test_grad(self, sin_curve):
        curve, interval = sin_curve
        x_grad, y_grad = curve.grad(interval)

        cos = np.cos(interval)

        assert pytest.approx(np.sum(np.abs(x_grad - np.ones_like(x_grad))), 0, 1e-6)
        assert pytest.approx(np.sum(np.abs(y_grad - cos)), 0, 1e-6)

    def test_grad_sq(self, sin_curve):
        curve, interval = sin_curve
        x_grad, y_grad = curve.grad_sq(interval)

        sin = -np.sin(interval)

        assert pytest.approx(np.sum(np.abs(x_grad - np.zeros_like(x_grad))), 0, 1e-6)
        assert pytest.approx(np.sum(np.abs(y_grad - sin)), 0, 1e-6)

    def test_tangent(self, sin_curve):
        curve, interval = sin_curve
        tangent = curve.tangent(interval)

        # Regular curve should not have any singularities
        assert len(np.where(np.sum(np.abs(tangent), 1) == 0)[0]) == 0

        # Tangent norm should always be 1
        assert np.sum(np.linalg.norm(tangent) - np.ones_like(tangent)) == 0

