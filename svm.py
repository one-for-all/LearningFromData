# Compare SVM and PLA on a linear classification problem
import numpy as np
from linear_problem_spec import Problem
from pla import PerceptronLearningAlgorithm


if __name__ == "__main__":
    N = 10
    n_trials = 1000
    pla = PerceptronLearningAlgorithm()
    w_initial = np.zeros(3)

    N_test = 10000

    for _ in range(n_trials):
        problem = Problem()
        x1s, x2s, ys = problem.sample(N)
        while np.sum(ys == ys[0]) == len(ys):
            x1s, x2s, ys = problem.sample(N)

        # Apply PLA
        w_pla, _ = pla.solve(x1s, x2s, ys, w_initial)

        # Apply SVM

        # test data
        x1s_test, x2s_test, ys_test = problem.sample(N_test)

        # Compute PLA out-sample error
        hypothesis_ys = pla.compute_multiple_labels(w_pla, x1s_test, x2s_test)
        pla_error = sum(hypothesis_ys != ys_test) / N_test