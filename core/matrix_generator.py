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


def generate_positive_definite_quadratic_form(dimensions, condition_number, eigenbasis_producer=None):
    """
    Generates a random positive definite quadratic form of given dimensions and condition number.
    The condition number is defined as the ratio of the greatest eigenvalue to the smallest one.

    The smallest eigenvalue is always 1.
    The greatest eigenvalue is always `condition_number`.
    The eigenvalues in between are random.

    Firstly the representation of the matrix in the eigenbasis is generated and then transformed to the canonical basis.
    """

    assert condition_number >= 1
    assert dimensions >= 2
    smallest_eigenvalue = 1.
    greatest_eigenvalue = smallest_eigenvalue * condition_number

    # The order of coordinates obviously doesn't matter
    eigenvalues = [smallest_eigenvalue] + [random() for _ in range(dimensions - 2)] + [greatest_eigenvalue]
    matrix_in_basis = np.diag(eigenvalues)

    B = eigenbasis_producer(dimensions)

    # Now we transform the matrix TO the canonical basis
    # Transposition matrix from canonical basis to B is just matrix `B`
    T_C_B = B
    T_B_C = np.linalg.inv(T_C_B)
    return QuadraticForm(T_C_B @ matrix_in_basis @ T_B_C)


def evaluator(computation: Callable[[_], int], n_times):
    pass
