from typing import Sequence, Callable

import numpy as np
import scipy.signal as signal


def explicit_2d_euler_step(u: np.ndarray, dt: float = 0.1, dx: float = 1.,
                           dy: float = 1.) -> np.ndarray:
    """
    Applies a single temporal Euler step based on the estimation of the central
    derivative of a 2D NumPy array.

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


def apply_euler_steps_central_derviative(u: np.ndarray, t: int = 1,
                                         dt: float = 0.1, dx: float = 1.,
                                         dy: float = 1.) -> Sequence[np.ndarray]:
    """
    Applies a t temporal Euler steps based on the estimation of the central
    derivatives to a 2D NumPy array.

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
        u_temp = explicit_2d_euler_step(u=u_t[-1].copy(), dt=dt, dx=dx, dy=dy)
        u_t.append(u_temp)

    return u_t


def heat_kernel(x: np.ndarray, n: int, t: float) -> np.ndarray:
    """
    Defines the heat heat_kernel operator at time t according to the equation:
    u(x, t) = (1 / (4 * pi * t)^(n / 2)) * exp((-1 / 4 * t) * x.T @ x)

    :param x: (int) array x over which to compute the heat_kernel
    :param n: (int) number of rows/columns in the heat heat_kernel
    :param t: (float) time-point t

    :return: (np.ndarray) The heat heat_kernel at time t with dimension n X n
    """

    # kernel = ((1 / np.power((4 * np.pi * t), (n / 2))) *
    #           np.exp((-1 / (4 * t)) * (x.T @ x)))

    kernel = - ((2 * np.pi) ** 2) * (x.T @ x) * t

    return kernel


def apply_2d_heat_kernel_step(u: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Applies a single step of the 2D heat equation via convolving the heat heat_kernel
    with U.

    :param u: (np.ndarray) An array containing U_t
    :param kernel: (np.ndarray) Heat heat_kernel to apply to U

    :return: (np.ndarray) An array containing U_t+1
    """

    f_im = fourier_transform(u)

    return signal.fftconvolve(u, kernel, 'same')


def gaussian(shape, kappa=1, t=.001):
    # Gaussian Kernel for the Heat kernel in Fourier domain
    n, k = shape
    center_i = int(n / 2)
    center_j = int(k / 2)

    # This image is symmetric so we do one quadrant and fill in the others
    kernel = np.zeros((n, k))
    for i in range(0, n - center_i):
        for j in range(0, k - center_j):
            temp = np.exp(-(i ** 2 + j ** 2) * kappa * t)
            kernel[center_i + i, center_j + j] = temp
            kernel[center_i - i, center_j + j] = temp
            kernel[center_i + i, center_j - j] = temp
            kernel[center_i - i, center_j - j] = temp

    return kernel


def fourier_transform(image):
    """
    Computes the Fourier transform on the image and shifts it to the "typical"
    representation that is shown.
    """

    temp = np.fft.fft2(image)

    # Remember to shift
    temp = np.fft.fftshift(temp)

    return temp


def inverse_fourier_transform(f_input_imag):
    imag = np.fft.ifftshift(f_input_imag)
    imag = np.fft.ifft2(imag)
    return np.abs(imag) ** 2  # Remove those imaginary values


def apply_2d_heat_kernel(u: np.ndarray, heat_kernel: Callable = heat_kernel,
                         t: int = 1, dt: float = 0.1) -> Sequence[np.ndarray]:
    """
    Applies t / dt steps of the 2D heat equation via convolving the heat kernel with U
    t / dt times.

    :param u: (np.ndarray) An array containing U_0
    :param heat_kernel: (Callable) a function which takes inputs x, n & t and returns a
    np.ndarray of shape [len(x), ] representing the heat kernel at time t over x.
    :param t: (int) The temporal length for which to apply steps to, defaults to 1
    :param dt: (float) The size of the temporal change, defaults to 0.1

    :return: (Sequence[np.ndarray]) A tuple, containing the arrays of U_1 to U_t
    """

    assert dt < t, f"dt must be smaller then t, but received dt = {dt}, and t = {t}"

    # Define the heat heat_kernel, we work in 2D
    n = 2
    m, k = u.shape

    # Start with U_0
    u_t = [u.copy(), ]
    current_t = dt

    # Define the grid for the heat heat_kernel
    # x1 = np.linspace(start=(m // 2), stop=0, num=(m // 2), dtype=np.int)
    # x1 = np.concatenate((x1[::-1], x1), 0)
    # x = np.expand_dims(x1, 0)
    # x = x / x.max()
    # x2 = np.linspace(start=(k // 2), stop=0, num=(k // 2), dtype=np.int)
    # x2 = np.concatenate((x2[::-1], x2), 0)
    # x2 = np.expand_dims(x2, 0)
    # x = x1.T @ x1
    # x = x / x.max()

    # Iterate over time
    steps = int(t / dt)
    for step in range(steps):
        # kernel = heat_kernel(x=x, n=n, t=current_t)
        kernel = gaussian(u.shape, kappa=4, t=current_t)
        u_temp = fourier_transform(u_t[-1].copy())
        u_temp = np.multiply(u_temp, kernel)
        u_temp = inverse_fourier_transform(u_temp.copy())

        # u_temp = apply_2d_heat_kernel_step(u=u_t[-1].copy(), kernel=kernel)
        current_t += dt
        u_t.append(u_temp)

    return u_t
