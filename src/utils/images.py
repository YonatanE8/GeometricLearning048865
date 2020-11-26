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


def heat_kernel(shape: tuple, t: float) -> np.ndarray:
    """
    Defines the heat heat_kernel operator at time t in the Fourier domain
    u = exp(-pi^2 * xi^T @ (4 * t * I) @ xi)

    :param shape: (tuple) a tuple of 2 ints denoting the shape of the image to which
     we apply the heat kernel
    :param n: (int) number of rows/columns in the heat heat_kernel
    :param t: (float) time-point t

    :return: (np.ndarray) The heat heat_kernel at time t with dimension shape
    """

    # Create the xi vectors
    x = np.linspace(start=(-shape[0] / 2), stop=(shape[0] / 2), num=shape[0])
    y = np.linspace(start=(-shape[1] / 2), stop=(shape[1] / 2), num=shape[1])
    x, y = np.meshgrid(x, y)

    # Compute the Fourier representation of the kernel
    kernel = np.exp(-np.power(np.pi, 2) * (np.power(x, 2) + np.power(y, 2)) * t)

    return kernel


def fourier_transform_im(image: np.ndarray) -> np.ndarray:
    """
    Computes the Fourier transform on the image and shifts it to the "typical"
    representation that is shown.

    :param image: (np.ndarray) A 2D numpy array, containing the image to transform

    :return: (np.ndarray) The transformed image
    """

    fft2d_im = np.fft.fft2(image)
    fft2d_im = np.fft.fftshift(fft2d_im)

    return fft2d_im


def inverse_fourier_transform(image: np.ndarray):
    """
    Computes the Inverse Fourier transform on the transformation of an image.
    starts by shifting it back.

    :param image: (np.ndarray) A 2D numpy array, containing the transformed image.

    :return: (np.ndarray) The absolute values of the inverse-transformed image
    (removes the imaginary part).
    """

    ifft2d_im = np.fft.ifftshift(image)
    ifft2d_im = np.fft.ifft2(ifft2d_im)

    return np.abs(ifft2d_im)


def apply_2d_heat_kernel_step(u: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Applies a single step of the 2D heat equation multiplying the heat kernel and the
     image in the Fourier domain and then transforms the image back.

    :param u: (np.ndarray) An array containing the image at time t
    :param kernel: (np.ndarray) the heat kernel at time t

    :return: (np.ndarray) An array containing the image at time t after applying
    the heat kernel
    """

    # Compute the Fourier transform of the image
    image = fourier_transform_im(u)

    # Apply the heat kernel in the spectral domain
    image = np.multiply(kernel, image)

    # Transform the image back
    image = inverse_fourier_transform(image)

    return image


def apply_2d_heat_kernel(u: np.ndarray, spectral_kernel: Callable = heat_kernel,
                         t: int = 1, dt: float = 0.1) -> Sequence[np.ndarray]:
    """
    Applies t / dt steps of the 2D heat equation via convolving the heat kernel with U
    t / dt times.

    :param u: (np.ndarray) An array containing U_0
    :param spectral_kernel: (Callable) a function which takes inputs 'shape' & t,
    and returns a np.ndarray of shape 'shape' representing the kernel in the
    spectral domain at time t.
    :param t: (int) The temporal length for which to apply steps to, defaults to 1
    :param dt: (float) The size of the temporal change, defaults to 0.1

    :return: (Sequence[np.ndarray]) A tuple, containing the arrays of U_1 to U_t
    """

    assert dt < t, f"dt must be smaller then t, but received dt = {dt}, and t = {t}"

    # Start with U_0
    u_t = [u.copy(), ]
    current_t = dt
    shape = u.shape

    # Iterate over time
    steps = int(t / dt)
    for step in range(steps):
        kernel = spectral_kernel(shape=shape, t=current_t)
        u_temp = apply_2d_heat_kernel_step(u=u_t[-1], kernel=kernel)
        u_t.append(u_temp.copy())
        current_t += dt

    return u_t


