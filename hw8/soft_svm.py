# Apply soft margin SVM on digits data
import numpy as np
from tqdm import tqdm
from qpsolvers import solve_qp


class SoftSVM:
    def __init__(self, C, Q):
        self.C = C
        self.Q = Q

    def solve(self, X, y):
        N = X.shape[0]

        K = np.power(np.matmul(X, X.T) + 1, self.Q)
        Y = np.matmul(y, y.T)

        P = np.multiply(K, Y)
        q = -np.ones(N)
        A = y.T.astype(float)
        b = np.array([[0.0]])
        lb = np.zeros(N)
        ub = self.C*np.ones(N)

        alpha = solve_qp(
            P=P, q=q, A=A, b=b, lb=lb, ub=ub, solver="cvxopt"
        )

        # Compute final hypothesis param values
        selector = ~np.isclose(alpha, 0, atol=1e-4)
        w = alpha[selector] * y.flatten()[selector]
        X_support = X[selector]
        x_s = X_support[:1]
        y_s = y.flatten()[0]
        K_s = np.power(np.matmul(x_s, X_support.T) + 1, self.Q)
        b = y_s - np.sum(w * K_s)

        return w, b, X_support

    def compute_y(self, w, b, X_support, X):
        K = np.power(np.matmul(X, X_support.T) + 1, self.Q)
        y = np.sum(w * K, axis=1) + b

        y[y > 0] = 1
        y[y <= 0] = -1
        return y


def error_rate(y1, y2):
    N = y1.shape[0]
    diff = y1 != y2
    return np.sum(y1 != y2) / N


if __name__ == "__main__":
    training_set = np.loadtxt("features.train")
    training_y = training_set[:200, :1].astype(int)
    training_X = training_set[:200, 1:]

    test_set = np.loadtxt("features.test")
    test_y = test_set[:, :1].astype(int)
    test_X = test_set[:, 1:]

    # Solve classification task of one digit vs others
    svm = SoftSVM(C=0.01, Q=2)
    target_digits = [0, 2, 4, 6, 8]
    E_in_s = []
    for target in tqdm(target_digits):
        target_y = np.zeros_like(training_y)
        target_y[training_y == target] = 1
        target_y[training_y != target] = -1

        w, b, X_support = svm.solve(training_X, target_y)
        hypo_y = svm.compute_y(w, b, X_support, training_X)
        E_in_s.append(error_rate(hypo_y, target_y.flatten()))

    for digit, E_in in zip(target_digits, E_in_s):
        print("E-insample for digit {} is {}".format(digit, E_in))
