import numpy as np
from tqdm import tqdm
from linear_problem_spec import Problem
from linear_regression import LinearRegression
from pla import PerceptronLearningAlgorithm
import matplotlib.pyplot as plt
from utils import draw_line, draw_points


def compute_error(w, x1s, x2s, ys):
    X = np.array([np.ones_like(x1s), x1s, x2s]).T
    y_pred = (np.matmul(X, w) > 0).astype(np.int8).reshape((-1,))
    y_pred[y_pred == 0] = -1
    error = np.sum(y_pred != ys)/len(ys)
    return error


if __name__ == "__main__":
    error_rate_list = []
    N_iters = 1000
    N_sample = 10
    E_in_list = []
    E_out_list = []
    pla_iters_list = []
    for _ in tqdm(range(N_iters)):
        p = Problem()
        x1s, x2s, ys = p.sample(num_points=N_sample)

        alg = LinearRegression()
        w_initial = alg.solve(x1s, x2s, ys)

        pla_alg = PerceptronLearningAlgorithm()
        w, pla_iters = pla_alg.solve(x1s, x2s, ys, w_initial.reshape((-1,)))
        pla_iters_list.append(pla_iters)

        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)
        # draw_line(plt, w, color='black')
        # draw_line(plt, p.w, color='green')
        # draw_points(plt, x1s, x2s, ys)
        # plt.show()
        # exit()

        # Compute in-sample error
        # E_in = compute_error(w, x1s, x2s, ys)
        # E_in_list.append(E_in)

        # Compute out-sample error
        # x1s_test, x2s_test, ys_test = p.sample(num_points=1000)
        # E_out = compute_error(w, x1s_test, x2s_test, ys_test)
        # E_out_list.append(E_out)

    print("avg pla iters: {}".format(np.mean(pla_iters_list)))

    # print("E in-sample: {}".format(np.mean(E_in_list)))
    # print("E out-sample: {}".format(np.mean(E_out_list)))
