# Linear regression algorithm
import numpy as np


class LinearRegression:
    def __init__(self):
        pass

    def solve(self, x1s, x2s, ys):
        # w = (X^T X)^(-1)X^T y
        X = np.array([np.ones_like(x1s), x1s, x2s]).T
        y = ys.reshape((-1, 1))
        w = np.matmul(np.linalg.pinv(X), y)
        return w


class NonlinearTransformation:
    def __init__(self):
        pass

    def solve(self, x1s, x2s, ys):
        # Nonlinear transformation to get features
        # 1, x1, x2, x1x2, x1**2, x2**2
        X = np.array([np.ones_like(x1s), x1s, x2s,
                      x1s*x2s, x1s**2, x2s**2]).T
        y = ys.reshape((-1, 1))
        w = np.matmul(np.linalg.pinv(X), y)
        return w
