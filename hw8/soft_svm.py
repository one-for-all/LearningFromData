# Apply soft margin SVM on digits data
import numpy as np
from tqdm import tqdm
from qpsolvers import solve_qp


class SoftSVM:
    def __init__(self, C, Q):
        self.C = C
        self.Q = Q

    def solve(self, X, y, K=None):
        N = X.shape[0]

        if K is None:
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
        selector = ~np.isclose(alpha, 0, atol=self.C*0.01)
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
    return np.sum(y1 != y2) / N


def construct_one_vs_five(X, y):
    one_indices = (y.flatten() == 1)
    five_indices = (y.flatten() == 5)

    one_X = X[one_indices]
    five_X = X[five_indices]
    target_X = np.vstack((one_X, five_X))

    N_one = one_X.shape[0]
    N = target_X.shape[0]

    target_y = np.ones((N, 1))
    target_y[N_one:] = -1

    return target_X, target_y


if __name__ == "__main__":
    # C = 0.01
    # Q = 2

    training_set = np.loadtxt("features.train")
    training_y = training_set[:, :1].astype(int)
    training_X = training_set[:, 1:]
    # training_K = np.power(np.matmul(training_X, training_X.T) + 1, Q)

    test_set = np.loadtxt("features.test")
    test_y = test_set[:, :1].astype(int)
    test_X = test_set[:, 1:]

    # # Solve classification task of one digit vs others
    # svm = SoftSVM(C=C, Q=Q)
    # target_digits = [0, 2, 4, 6, 8]
    # E_in_s = []
    # num_svs = []
    # for target in tqdm(target_digits):
    #     target_y = np.zeros_like(training_y)
    #     target_y[training_y == target] = 1
    #     target_y[training_y != target] = -1
    #
    #     w, b, X_support = svm.solve(training_X, target_y, K=training_K)
    #     hypo_y = svm.compute_y(w, b, X_support, training_X)
    #     E_in_s.append(error_rate(hypo_y, target_y.flatten()))
    #     num_svs.append(len(w))
    #
    # for digit, E_in, num_sv in zip(target_digits, E_in_s, num_svs):
    #     print("E-insample for digit {} is {}, with {} support vectors".format(digit, E_in, num_sv))

    # Construct 1 vs 5 dataset
    train_X, train_y = construct_one_vs_five(training_X, training_y)
    test_X, test_y = construct_one_vs_five(test_X, test_y)

    # C_values = [0.001, 0.01, 0.1, 1]
    C = 1
    Q_values = [2, 5]
    # Q = 2

    # train_K = np.power(np.matmul(train_X, train_X.T) + 1, Q)

    E_in_s = []
    E_out_s = []
    num_svs = []
    for Q in tqdm(Q_values):
        svm = SoftSVM(C=C, Q=Q)
        w, b, X_support = svm.solve(train_X, train_y, K=None)
        hypo_y_in = svm.compute_y(w, b, X_support, train_X)
        E_in_s.append(error_rate(hypo_y_in, train_y.flatten()))
        hypo_y_out = svm.compute_y(w, b, X_support, test_X)
        E_out_s.append(error_rate(hypo_y_out, test_y.flatten()))
        num_svs.append(len(w))

    print("C = {}".format(C))
    for Q, E_in, E_out, num_sv in zip(Q_values, E_in_s, E_out_s, num_svs):
        print("Q: {}, E_in: {}, E_out: {}, num sv: {}".format(Q, E_in, E_out, num_sv))
