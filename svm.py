# Compare SVM and PLA on a linear classification problem
import numpy as np
from linear_problem_spec import Problem
from pla import PerceptronLearningAlgorithm
from tqdm import tqdm
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
from utils import draw_line, draw_points


class SVM:
    def __init__(self):
        pass

    def solve(self, x1s, x2s, ys):
        N_points = len(x1s)

        P = np.zeros([3, 3])
        P[1:, 1:] = np.identity(2)
        q = np.zeros(3)
        G = np.zeros([N_points, 3])
        G[:, 0] = -ys
        G[:, 1] = -ys*x1s
        G[:, 2] = -ys*x2s
        h = -np.ones(N_points)

        w = solve_qp(P, q, G, h, solver="cvxopt")

        if w is None:
            return None, None

        # Compute number of support vectors
        constraints = np.matmul(-G, w.reshape([3, 1]))
        num_support_vectors = np.sum(np.isclose(constraints, 1, rtol=1e-5))

        return w, num_support_vectors


if __name__ == "__main__":
    N = 100
    n_trials = 1000
    pla = PerceptronLearningAlgorithm()
    svm = SVM()
    w_initial = np.zeros(3)

    N_test = 10000

    # Keep track of whether svm better
    svm_better = []
    num_svs = []

    for _ in tqdm(range(n_trials)):
        problem = Problem()
        x1s, x2s, ys = problem.sample(N)
        while np.sum(ys == ys[0]) == len(ys):
            x1s, x2s, ys = problem.sample(N)

        # Apply SVM
        w_svm, num_sv = svm.solve(x1s, x2s, ys)
        while w_svm is None:
            x1s, x2s, ys = problem.sample(N)
            while np.sum(ys == ys[0]) == len(ys):
                x1s, x2s, ys = problem.sample(N)
            w_svm, num_sv = svm.solve(x1s, x2s, ys)

        # Apply PLA
        w_pla, _ = pla.solve(x1s, x2s, ys, w_initial)

        # test data
        x1s_test, x2s_test, ys_test = problem.sample(N_test)

        # Compute PLA out-sample error
        pla_ys = pla.compute_multiple_labels(w_pla, x1s_test, x2s_test)
        pla_error = np.sum(pla_ys != ys_test) / N_test

        # Compute SVM out-sample error
        svm_ys = pla.compute_multiple_labels(w_svm, x1s_test, x2s_test)
        svm_error = np.sum(svm_ys != ys_test) / N_test

        svm_better.append(svm_error < pla_error)
        num_svs.append(num_sv)

        # # For plotting samples, true line and proposed line
        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)
        # draw_line(plt, w_svm, color='black')
        # draw_line(plt, problem.w, color='green')
        # draw_points(plt, x1s, x2s, ys)
        # plt.savefig("plot.png")

    svm_better_percentage = np.sum(svm_better) / n_trials
    average_svs = np.mean(num_svs)
    print("Percentage of time SVM better: {}%".format(svm_better_percentage*100))
    print("Average number of support vectors: {}".format(average_svs))
