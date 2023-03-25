from typing import Tuple, NamedTuple

import numpy as np
from matplotlib import pyplot as plt
from numpy import newaxis


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


def visualize_optimizing_process(f, roi: SearchRegion2d, points: np.ndarray[np.ndarray[float]]):
    X, Y = auto_meshgrid(f, roi)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(f(points.T))
    ax1.grid()
    ax2.plot(points[:, 0], points[:, 1], 'o-')
    print(f"Optimizer trajectory:")
    print(points)
    print(f"Best value found: x* = {points[-1]} with f(x*) = {f(points[-1])}")
    levels = np.concatenate((f(points.T), np.linspace(-1, 1, 100)))
    ax2.contour(X, Y, f(np.stack((X, Y))), levels=sorted(set(levels)))
    # ax2.contour(X, Y, f([X, Y]), levels=sorted(set([f(p) for p in points])))
