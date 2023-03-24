from typing import Callable, NamedTuple

import numpy as np
from numpy import newaxis


class QuadraticForm(NamedTuple):
    matrix: np.ndarray

    def __call__(self, x: np.ndarray):
        return x[:, newaxis] @ self.matrix @ x

def evaluator(computation: Callable[[_], int], n_times):
    pass
