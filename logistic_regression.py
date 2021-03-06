# Perform logistic regression
import numpy as np
from linear_problem_spec import Problem
from tqdm import tqdm


class LogisticRegression:
    def __init__(self):
        pass

    def solve(self, x1s, x2s, ys):
        num_points = len(x1s)
        eta = 0.01
        w = np.ones(3)
        w_next = np.zeros(3)
        epochs = 0
        while np.linalg.norm(w_next-w) >= 0.01:
            w = np.copy(w_next)
            permutation = np.random.permutation(num_points)
            for idx in permutation:
                x1, x2, y = x1s[idx], x2s[idx], ys[idx]
                x = np.array([1, x1, x2])
                gradient = -y*x / (1 + np.exp(y*np.dot(w_next, x)))
                w_next -= eta*gradient
            epochs += 1
        return w_next, epochs


def estimate_Eout(w, problem):
    N = 1000
    x1s, x2s, ys = problem.sample(N)
    X = np.array([np.ones_like(x1s), x1s, x2s]).T
    logit = np.matmul(X, w.reshape((-1, 1))).reshape((-1,))
    assert logit.shape == x1s.shape
    Eout = 1/N * np.sum(np.log(1 + np.exp(-ys*logit)))
    return Eout


if __name__ == "__main__":
    num_runs = 100
    N = 100
    logistic_regression = LogisticRegression()
    Eout_values = []
    epoch_values = []
    for _ in tqdm(range(num_runs)):
        p = Problem()
        x1s, x2s, ys = p.sample(N)
        w, epochs = logistic_regression.solve(x1s, x2s, ys)
        Eout = estimate_Eout(w, p)
        Eout_values.append(Eout)
        epoch_values.append(epochs)

    print("Eout: {}".format(np.mean(Eout_values)))
    print("epochs: {}".format(np.mean(epoch_values)))
