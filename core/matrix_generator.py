from random import random
from typing import Callable, NamedTuple

import numpy as np
from numpy import newaxis


class QuadraticForm(NamedTuple):
    matrix: np.ndarray

    def __call__(self, x: np.ndarray):
        return x @ self.matrix @ x[:, newaxis]


def random_basis(n):
    pass


def canonical_basis(n):
    return np.eye(n)


def generate_positive_definite_quadratic_form(dimensions, condition_number, basis_producer=None):
    assert condition_number >= 1
    assert dimensions >= 2
    smallest_eigenvalue = 1.
    greatest_eigenvalue = smallest_eigenvalue * condition_number

    eigenvalues = [smallest_eigenvalue] + [random() for _ in range(dimensions - 2)] + [greatest_eigenvalue]


def evaluator(computation: Callable[[_], int], n_times):
    pass
