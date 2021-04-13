# Apply soft margin SVM on digits data
import numpy as np
from tqdm import tqdm
from qpsolvers import solve_qp


class Kernel:
    POLYNOMIAL = "polynomial"
    RBF = "radial basis function"


class Backend:
    QPSOLVER = "qpsolver"
    SKLERAN = "sklearn"


class SoftSVM:
    def __init__(self, C, Q, kernel=Kernel.POLYNOMIAL, backend=Backend.QPSOLVER):
        """
        :param flaot C: Parameter that controls the softness of the SVM.
            Larger value allows less violation of the margin.
        :param int Q: Degree of the polynomial kernel. Ignored if kernel is not polynomial
        :param str kernel: Kernel of the support vector.
        :param str backend: Backend used for solving the SVM.
        """
        self.C = C
        self.Q = Q
        self.kernel = kernel
        self.backend = backend

        self.w = None
        self.b = None
        self.X_support = None

    def solve(self, X, y, K=None):
        """
        :param X: Input matrix of shape (N, d)
        :param y: Target matrix of shape (N, 1), each value being +1 or -1.
        :param K: Pre-computed kernel matrix or None
        :return:
        """
        N, d = X.shape

        # Construct kernel matrix
        if K is None:
            if self.kernel == Kernel.POLYNOMIAL:
                K = np.power(np.matmul(X, X.T) + 1, self.Q)
            else:
                raise NotImplementedError

        # Quadratic matrix
        Y = np.matmul(y, y.T)
        P = np.multiply(K, Y)

        # Linear matrix
        q = -np.ones(N)

        # Equality constraint matrix
        A = y.T.astype(float)

        # Equality target matrix
        b = np.array([[0.0]])

        # Lower-bound and upper-bound on the variables
        lb = np.zeros(N)
        ub = self.C*np.ones(N) if self.C else None

        # Solve the QP program
        alpha = solve_qp(
            P=P, q=q, A=A, b=b, lb=lb, ub=ub, solver="cvxopt"
        )

        # Compute final hypothesis w and support vectors
        if self.C:
            selector = ~np.isclose(alpha, 0, atol=min(self.C*0.01, 1e-4))
        else:
            selector = ~np.isclose(alpha, 0, atol=1e-4)
        w = alpha[selector] * y.flatten()[selector]
        X_support = X[selector]

        # Compute b
        x_s = X_support[:1, :]
        assert x_s.shape == (1, d)
        y_s = y.flatten()[0]
        K_s = np.power(np.matmul(x_s, X_support.T) + 1, self.Q)
        b = y_s - np.sum(w * K_s)

        self.w = w
        self.b = b
        self.X_support = X_support

    def compute_y(self, X):
        if self.w is None:
            raise ValueError

        w, b, X_support, = self.w, self.b, self.X_support
        K = np.power(np.matmul(X, X_support.T) + 1, self.Q)
        y = np.sum(w * K, axis=1) + b

        y[y > 0] = 1
        y[y <= 0] = -1
        return y.reshape((-1, 1))

    @property
    def num_support_vectors(self):
        if self.w is None:
            raise ValueError
        return len(self.w)


def error_rate(y1, y2):
    N = y1.shape[0]
    assert y1.shape == y2.shape == (N, 1)
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


def digit_vs_all(training_X, training_y, test_X, test_y, digit, C, Q, training_K=None):
    # Construct the target y matrix for one digit vs all others
    target_y = np.zeros_like(training_y)
    target_y[training_y == digit] = 1
    target_y[training_y != digit] = -1

    # Solve SVM
    svm = SoftSVM(C=C, Q=Q, kernel=Kernel.POLYNOMIAL, backend=Backend.QPSOLVER)
    svm.solve(training_X, target_y, K=training_K)

    # Compute in-sample error rate
    hypo_y_in = svm.compute_y(training_X)
    E_in = error_rate(hypo_y_in, target_y)

    # Construct the target y matrix for one digit vs all others
    target_test_y = np.zeros_like(test_y)
    target_test_y[test_y == digit] = 1
    target_test_y[test_y != digit] = -1

    # Compute out-sample error rate
    hypo_y_out = svm.compute_y(test_X)
    E_out = error_rate(hypo_y_out, target_test_y)

    # Get number of support vectors
    num_svs = svm.num_support_vectors

    print("E_in: {:.3f}\nE_out: {:.3f}\nnumber of support vectors: {}".format(E_in, E_out, num_svs))


if __name__ == "__main__":
    # Load training set
    training_set = np.loadtxt("features.train")
    training_y = training_set[:200, :1].astype(int)
    training_X = training_set[:200, 1:]

    # Load test set
    test_set = np.loadtxt("features.test")
    test_y = test_set[:, :1].astype(int)
    test_X = test_set[:, 1:]

    # Parameters
    target_digits = [2]
    C = 100
    Q = 3

    # Pre-compute kernel matrix to save time
    training_K = np.power(np.matmul(training_X, training_X.T) + 1, Q)

    for digit in tqdm(target_digits):
        print("digit {} with backend {}:".format(digit, Backend.QPSOLVER))
        digit_vs_all(training_X, training_y, test_X, test_y, digit, C, Q, training_K=training_K)
        print("==============")

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

    # # Construct 1 vs 5 dataset
    # train_X, train_y = construct_one_vs_five(training_X, training_y)
    # test_X, test_y = construct_one_vs_five(test_X, test_y)
    #
    # # C_values = [0.001, 0.01, 0.1, 1]
    # C = 1
    # Q_values = [2, 5]
    # # Q = 2
    #
    # # train_K = np.power(np.matmul(train_X, train_X.T) + 1, Q)
    #
    # E_in_s = []
    # E_out_s = []
    # num_svs = []
    # for Q in tqdm(Q_values):
    #     svm = SoftSVM(C=C, Q=Q)
    #     w, b, X_support = svm.solve(train_X, train_y, K=None)
    #     hypo_y_in = svm.compute_y(w, b, X_support, train_X)
    #     E_in_s.append(error_rate(hypo_y_in, train_y.flatten()))
    #     hypo_y_out = svm.compute_y(w, b, X_support, test_X)
    #     E_out_s.append(error_rate(hypo_y_out, test_y.flatten()))
    #     num_svs.append(len(w))
    #
    # print("C = {}".format(C))
    # for Q, E_in, E_out, num_sv in zip(Q_values, E_in_s, E_out_s, num_svs):
    #     print("Q: {}, E_in: {}, E_out: {}, num sv: {}".format(Q, E_in, E_out, num_sv))
