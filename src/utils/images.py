from typing import Sequence
import numpy as np


def explicit_2d_heat_equation_step(u: np.ndarray, dt: float = 0.1, dx: float = 1.,
                                   dy: float = 1.) -> np.ndarray:
    """
    Applies a single temporal step of the 2D heat equation to a 2D NumPy array.

    :param u: (np.ndarray) An array containing U_t
    :param dt: (float) The size of the temporal change, defaults to 0.1
    :param dx: (float) The size of the change in the X axis, defaults to 1.0
    :param dy: (float) The size of the change in the Y axis, defaults to 1.0

    :return: (np.ndarray) An array containing U_t+1
    """

    # Creat copy of arrays to avoid in in-place changes related errors
    u_xx = u.copy()
    u_yy = u.copy()

    # Compute U_xx & U_yy
    u_xx[:, 1:-1] = (u[:, 2:] - (2 * u[:, 1:-1]) + u[:, :-2]) / (dx ** 2)
    u_yy[1:-1, :] = (u[2:, :] - (2 * u[1:-1, :]) + u[:-2, :]) / (dy ** 2)

    # Handle boundaries
    u_xx[:, 0] = (u[:, 1] - (2 * u[:, 0])) / (dx ** 2)
    u_yy[0, :] = (u[1, :] - (2 * u[0, :])) / (dy ** 2)
    u_xx[:, -1] = (-(2 * u[:, -1]) + u[:, -2]) / (dx ** 2)
    u_yy[-1, :] = (-(2 * u[-1, :]) + u[-2, :]) / (dy ** 2)

    # Compute U_t+1
    u_next = u + dt * (u_xx + u_yy)

    return u_next


def apply_explicit_2d_heat_equation(u: np.ndarray, t: int = 1,
                                    dt: float = 0.1, dx: float = 1.,
                                    dy: float = 1.) -> Sequence[np.ndarray]:
    """
    Applies a t temporal steps of the 2D heat equation to a 2D NumPy array.

    :param u: (np.ndarray) An array containing U_0
    :param t: (int) The temporal length for which to apply steps to, defaults to 1
    :param dt: (float) The size of the temporal change, defaults to 0.1
    :param dx: (float) The size of the change in the X axis, defaults to 1.0
    :param dy: (float) The size of the change in the Y axis, defaults to 1.0

    :return: (Sequence[np.ndarray]) A tuple, containing the arrays of U_1 to U_t
    """

    assert dt < t, f"dt must be smaller then t, but received dt = {dt}, and t = {t}"

    # Start with U_0
    u_t = [u.copy(), ]

    # Iterate over time
    steps = int(t / dt)
    for step in range(steps):
        u_temp = explicit_2d_heat_equation_step(u=u_t[-1].copy(), dt=dt, dx=dx, dy=dy)
        u_t.append(u_temp)

    return u_t
