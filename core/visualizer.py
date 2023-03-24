from typing import Tuple, NamedTuple

import numpy as np
from matplotlib import pyplot as plt


class SearchRegion2d(NamedTuple):
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]


def debug_mesh(roi: SearchRegion2d):
    return np.meshgrid(
        np.linspace(roi.x_range[0], roi.x_range[1], 1000),
        np.linspace(roi.y_range[0], roi.y_range[1], 1000)
    )


def visualize_function_3d(f, roi: SearchRegion2d):
    X, Y = debug_mesh(roi)
    ax = plt.figure().add_subplot(projection='3d')
    return ax.plot_surface(X, Y, f(np.stack((X, Y))))


def visualize_optimizing_process(f, roi: SearchRegion2d, points):
    X, Y = debug_mesh(roi)
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

