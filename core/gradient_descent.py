import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from numpy import newaxis

precision = 1e-5


def gradient_descent(target_function, gradient_function, x0, linear_search, terminate_condition):
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


def find_upper_bound(f):
    original = f(0)
    r = 1
    while f(r) < original:
        r *= 2
    return r


def fixed_step_search(step_length):
    return lambda f, derivative: step_length # * derivative(0)


def bin_search(f, derivative):
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


def golden_ratio_search(f, _derivative):
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

# TODO: n-ary search through log space?


def symmetric_gradient_computer(f, h=precision):
    def computer(x):
        n = x.size
        return (f(x[:, newaxis] + h * np.eye(n)) - f(x[:, newaxis] - h * np.eye(n))) / (2 * h)

    return computer
