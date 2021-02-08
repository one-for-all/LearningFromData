# Nonlinear transformation on circular data
import numpy as np
from circle_problem_spec import Problem
from utils import draw_points
import matplotlib.pyplot as plt
from tqdm import tqdm
from linear_regression import (
    LinearRegression,
    NonlinearTransformation
)


def compute_error(w, x1s, x2s, ys):
    X = np.array([np.ones_like(x1s), x1s, x2s]).T
    y_pred = (np.matmul(X, w) > 0).astype(np.int8).reshape((-1,))
    y_pred[y_pred == 0] = -1
    error = np.sum(y_pred != ys)/len(ys)
    return error


def compute_error_nonlinear(w, x1s, x2s, ys):
    X = np.array([np.ones_like(x1s), x1s, x2s,
                  x1s * x2s, x1s ** 2, x2s ** 2]).T
    y_pred = (np.matmul(X, w) > 0).astype(np.int8).reshape((-1,))
    y_pred[y_pred == 0] = -1
    error = np.sum(y_pred != ys) / len(ys)
    return error


if __name__ == "__main__":
    N_iters = 1000
    N_samples = 1000
    E_in_list = [[] for _ in range(6)]
    E_out_list = []

    w_compare = np.array([
        [-1, -0.05, 0.08, 0.13, 1.5, 1.5],
        [-1, -0.05, 0.08, 0.13, 1.5, 15],
        [-1, -0.05, 0.08, 0.13, 15, 1.5],
        [-1, -1.5, 0.08, 0.13, 0.05, 0.05],
        [-1, -0.05, 0.08, 1.5, 0.15, 0.15]
    ])

    for _ in tqdm(range(N_iters)):
        p = Problem()
        x1s, x2s, ys = p.sample(num_points=N_samples)

        # alg = LinearRegression()
        # w = alg.solve(x1s, x2s, ys)

        # Compute in-sample error
        # E_in = compute_error(w, x1s, x2s, ys)
        # E_in_list.append(E_in)

        alg = NonlinearTransformation()
        w = alg.solve(x1s, x2s, ys)

        # Compute in-sample errors
        E_in = compute_error_nonlinear(w, x1s, x2s, ys)
        E_in_list[0].append(E_in)
        for idx, w_c in enumerate(w_compare):
            E_in = compute_error_nonlinear(w_c.reshape((-1, 1)), x1s, x2s, ys)
            E_in_list[idx+1].append(E_in)

        # Compute out-sample error
        x1s_test, x2s_test, ys_test = p.sample(num_points=1000)
        E_out = compute_error_nonlinear(w, x1s_test, x2s_test, ys_test)
        E_out_list.append(E_out)

    print("E in-sample values:")
    for idx in range(6):
        print(np.mean(E_in_list[idx]))

    print("E out-sample: {}".format(np.mean(E_out_list)))

    # print("E in-sample: {}".format(np.mean(E_in_list)))
