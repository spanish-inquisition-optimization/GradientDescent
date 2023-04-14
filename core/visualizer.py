from typing import Tuple, NamedTuple, Callable

import numpy as np
from matplotlib import pyplot as plt
from numpy import newaxis

from core.gradient_descent import symmetric_gradient_computer, gradient_descent


class SearchRegion2d(NamedTuple):
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]


def debug_mesh(roi: SearchRegion2d, n=1000):
    return np.meshgrid(
        np.linspace(roi.x_range[0], roi.x_range[1], n),
        np.linspace(roi.y_range[0], roi.y_range[1], n)
    )


def partially_vectorize(f, f_input_dims):
    def vectorized_f(t):
        if len(t.shape) == f_input_dims:
            return f(t)
        else:
            splitted = np.split(t, t.shape[-1], axis=-1)
            slices_along_last_axis = [splitted[i][..., 0] for i in range(t.shape[-1])]
            return np.concatenate([vectorized_f(s)[..., newaxis] for s in slices_along_last_axis], axis=-1)

    return vectorized_f


def supports_argument(f, smple_arg):
    try:
        f(smple_arg)
        return True
    except:
        return False


class AutoVectorizedFunction:
    def __init__(self, f, f_input_dims=None):
        self.f = f
        self.f_input_dims = f_input_dims

    def __call__(self, t):
        try:
            return self.f(t)
        except:
            assert self.f_input_dims is not None
            return partially_vectorize(self.f, self.f_input_dims)(t)


def auto_meshgrid(f, roi: SearchRegion2d):
    X, Y = debug_mesh(roi, 1000) if supports_argument(f, np.stack(debug_mesh(roi, 1))) else debug_mesh(roi, 300)
    return X, Y


def visualize_function_3d(f, roi: SearchRegion2d):
    X, Y = auto_meshgrid(f, roi)
    ax = plt.figure().add_subplot(projection='3d')
    return ax.plot_surface(X, Y, AutoVectorizedFunction(f, 1)(np.stack((X, Y))))


def visualize_optimizing_process(f, roi: SearchRegion2d, points, true_minimum=None):
    X, Y = auto_meshgrid(f, roi)
    vectorized_f = AutoVectorizedFunction(f, 1)

    if true_minimum is None:
        fig, (ax1, ax3) = plt.subplots(1, 2)
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.plot(vectorized_f(points.T))
    ax1.set_title(f"Function value")
    ax1.grid()

    if true_minimum is not None:
        ax2.plot(vectorized_f(points.T) - true_minimum)
        ax2.set_yscale("log", nonpositive='clip')
        ax2.set_title(f"Logarithmic error")
        ax2.grid()

    ax3.plot(points[:, 0], points[:, 1], 'o-')
    print(f"Optimizer trajectory:")
    print(points)
    print(f"Best value found: x* = {points[-1]} with f(x*) = {vectorized_f(points[-1])}")

    levels = vectorized_f(points.T)
    ax3.contour(X, Y, vectorized_f(np.stack((X, Y))), levels=sorted(set(levels)))
    ax3.set_title(f"Visited contours")

    return fig  # For further customization


def plot_section_graphs(f, discrete_param_values, continuous_param_values):
    fig, plots = plt.subplots(1, len(discrete_param_values))
    for i, discrete_param_value in enumerate(discrete_param_values):
        plots[i].plot(continuous_param_values, [f(discrete_param_value, x) for x in continuous_param_values])
        plots[i].set_title(f"Section at {discrete_param_value}")
        plots[i].set_yscale("log")
        plots[i].set_xscale("log")

        plots[i].grid()


def test_linear_search(linear_search: Callable[[Callable[[float], float], Callable[[float], float]], float]):
    alpha = 5

    def quadratic(x):
        return alpha * (x[0] - 5) ** 2 + (x[1] - 7) ** 2

    def quadratic_derivative(x):
        return np.array([2 * alpha * (x[0] - 5), 2 * (x[1] - 7)])

    quadratic_roi = SearchRegion2d((-50, 50), (-50, 50))

    def trig(x):
        return np.sin(0.5 * x[0] ** 2 - 0.25 * x[1] ** 2 + 3) * np.cos(2 * x[0] + 1 - np.exp(x[1]))

    trig_derivative = symmetric_gradient_computer(trig)
    trig_roi = SearchRegion2d((-1.5, 1.5), (-1.5, 2))

    from scipy.optimize import rosen, rosen_der as rosen_derivative
    rosen_roi = SearchRegion2d((-2, 1), (-1, 1))

    class SearchSetup(NamedTuple):
        f: Callable[[np.ndarray], float]
        gradient: Callable[[np.ndarray], np.ndarray]
        roi: SearchRegion2d
        num_steps: int
        start_point: np.ndarray
        true_minimum: float = None

    setups = [SearchSetup(quadratic, quadratic_derivative, quadratic_roi, 50, np.array([-20, -20]), 0),
              SearchSetup(trig, trig_derivative, trig_roi, 300, np.array([-0.1, -0.4])),
              SearchSetup(rosen, rosen_derivative, rosen_roi, 50, np.array([-1.5, -1.5]), 0)
            ]

    for setup in setups:
        visualize_function_3d(setup.f, setup.roi)
        visualize_optimizing_process(setup.f, setup.roi,
                                     np.array(gradient_descent(
                                         setup.f,
                                         setup.gradient,
                                         setup.start_point,
                                         linear_search,
                                         lambda f, points: len(points) > setup.num_steps)), setup.true_minimum)
