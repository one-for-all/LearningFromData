# Script for running experiment
import matplotlib.pyplot as plt
import numpy as np

from problem_spec import Problem
from pla import PerceptronLearningAlgorithm


def draw_line(plt, w, color='g'):
    x1s = np.linspace(-1, 1, 10)
    x2s = (-w[0] - w[1] * x1s) / w[2]
    plt.plot(x1s, x2s, color=color)


def draw_points(plt, x1s, x2s, ys):
    for x1, x2, y in zip(x1s, x2s, ys):
        plt.scatter(x1, x2, color=('b' if y == 1 else 'r'))


if __name__ == "__main__":
    n_iters_list = []
    error_rate_list = []
    for _ in range(1000):
        p = Problem()
        x1s, x2s, ys = p.sample(num_points=10)

        alg = PerceptronLearningAlgorithm()
        w, n_iters = alg.solve(x1s, x2s, ys)
        n_iters_list.append(n_iters)

        # Compute error probability
        num_samples = 1000
        test_x1s, test_x2s, test_ys = p.sample(num_points=num_samples)
        hypothesis_ys = (w[0] + w[1] * test_x1s + w[2] * test_x2s > 0).astype(np.int8)
        hypothesis_ys[hypothesis_ys == 0] = -1
        error_rate = sum(hypothesis_ys != test_ys) / num_samples
        error_rate_list.append(error_rate)

    print("avg iters: {}".format(np.mean(n_iters_list)))
    print("avg error rate: {}".format(np.mean(error_rate_list)))

    # For plotting samples, true line and proposed line
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    # draw_line(plt, w, color='black')
    # draw_line(plt, p.w, color='green')
    # draw_points(plt, x1s, x2s, ys)
    # plt.show()