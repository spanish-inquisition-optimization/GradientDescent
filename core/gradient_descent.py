from typing import Callable, List

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from numpy import newaxis

precision = 1e-5


def gradient_descent(target_function: Callable[[np.ndarray], float],
                     gradient_function: Callable[[np.ndarray], np.ndarray],
                     x0: np.ndarray,
                     linear_search: Callable[[Callable[[float], float], Callable[[float], float]], float],
                     terminate_condition: Callable[[Callable[[np.ndarray], float], List[np.ndarray]], bool]):
    points = [np.array(x0)]
    while not terminate_condition(target_function, points):
        last_point = points[-1]
        g = np.array(gradient_function(last_point))
        if np.linalg.norm(g) == 0:
            return points
        next_point = last_point - g * linear_search(lambda l: target_function(last_point - g * l),
                                                    lambda l: -np.dot(g, gradient_function(last_point - g * l)))
        points.append(next_point)
    return points


def find_upper_bound(f: Callable[[float], float]):
    original = f(0)
    r = 1
    while f(r) < original:
        r *= 2
    return r


def fixed_step_search(step_length):
    return lambda f, derivative: step_length


def bin_search(f: Callable[[float], float], derivative: Callable[[float], float]):
    # assume derivative(0) < 0 and derivative is rising
    l = 0
    r = find_upper_bound(f)

    while r - l > precision:
        m = (l + r) / 2
        if derivative(m) < 0:
            l = m
        else:
            r = m
    return r


def bin_search_with_iters(iters):
    def search(f: Callable[[float], float], derivative: Callable[[float], float]):
        # assume derivative(0) < 0 and derivative is rising
        i = 0
        l = 0
        r = find_upper_bound(f)

        while i < iters and r - l > precision and abs(derivative(r)) > precision:
            m = (l + r) / 2
            i += 1
            if derivative(m) < 0:
                l = m
            else:
                r = m
        return r

    return search


def golden_ratio_search(f: Callable[[float], float], _derivative: Callable[[float], float]):
    l = 0
    r = find_upper_bound(f)

    while r - l > precision:
        delta = (r - l) / scipy.constants.golden
        x1, x2 = r - delta, l + delta
        if f(x1) < f(x2):
            r = x2
        else:
            l = x1
    return r


def fibonacci_search(n_iters):
    def search(f, _derivative):
        l = 0
        r = find_upper_bound(f)
        length = r - l
        fibs = [1, 1]
        while len(fibs) <= n_iters:
            fibs.append(fibs[-1] + fibs[-2])
        x1 = l + length * fibs[-3] / fibs[-1]
        x2 = l + length * fibs[-2] / fibs[-1]
        y1, y2 = f(x1), f(x2)
        for k in range(n_iters - 2):
            if f(x1) > f(x2):
                l = x1
                x1 = x2
                x2 = l + (r - l) * fibs[-k - 3] / fibs[-k - 2]
                y1, y2 = y2, f(x2)
            else:
                r = x2
                x2 = x1
                x1 = l + (r - l) * fibs[-k - 4] / fibs[-k - 2]
                y1, y2 = f(x1), y1

        return r

    return search


def wolfe_conditions_search(c1, c2):
    assert 0 < c1 < c2 < 1

    def search(f: Callable[[float], float], derivative: Callable[[float], float]):
        # Need to find x such that:
        # 1) f(x) <= f(0) + c1 * x * derivative(0)
        # 2) derivative(x) >= c2 * derivative(0)
        initial_value = f(0)
        initial_slope = derivative(0)
        desired_slope = initial_slope * c2

        def desired_descent(x):
            return initial_value + initial_slope * c1 * x

        left = 0
        right = find_upper_bound(lambda x: f(x) - desired_descent(x) + initial_value)
        while right - left > precision:
            mid = (left + right) / 2
            if derivative(mid) < desired_slope:
                left = mid
            else:
                right = mid
                if f(mid) <= desired_descent(mid):
                    break
        return right

    return search


# TODO: n-ary search through log space?


def precision_termination_condition(_target_function: Callable[[np.ndarray], float], points: List[np.ndarray]):
    return len(points) > 2 and np.linalg.norm(points[-1] - points[-2]) < precision


def coordinate_vector_like(coordinate_index: int, reference: np.ndarray):
    res = np.zeros_like(reference)
    res[coordinate_index] = 1
    return res


def symmetric_gradient_computer(f: Callable[[np.ndarray], float], h: float = precision):
    def computer(x):
        # This trick only works on functions defined
        # in terms of scalar (or dimension-independent) np operations (aka ufuncs) which can thus be vectorized…
        # return (f(x[:, newaxis] + h * np.eye(n)) - f(x[:, newaxis] - h * np.eye(n))) / (2 * h)

        return np.array([
            (f(x + h * coordinate_vector_like(i, x)) - f(x - h * coordinate_vector_like(i, x))) / (2 * h)
            for i in range(x.size)
        ])

    return computer
