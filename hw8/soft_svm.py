# Apply soft margin SVM on digits data
import numpy as np
from tqdm import tqdm
from qpsolvers import solve_qp
from libsvm.python.svmutil import *


class Kernel:
    POLYNOMIAL = "polynomial"
    RBF = "radial basis function"


class Backend:
    QPSOLVER = "qpsolver"
    SKLERAN = "sklearn"
    LIBSVM = "libsvm"


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

    def fit(self, X, y, K=None, cross_validation=False):
        """
        :param X: Input matrix of shape (N, d)
        :param y: Target matrix of shape (N, 1), each value being +1 or -1.
        :param K: Pre-computed kernel matrix or None
        :return:
        """
        if self.backend == Backend.LIBSVM:
            if self.kernel == Kernel.POLYNOMIAL:
                kernel_type = 1
                args = '-t {} -d {} -g 1 -r 1 -c {} -q'.format(kernel_type, self.Q, self.C)
            elif self.kernel == Kernel.RBF:
                kernel_type = 2
                args = '-t {} -g 1 -r 1 -c {} -q'.format(kernel_type, self.C)
            else:
                raise ValueError
            if cross_validation:
                args += " -v 10"
                cv_accuracy = svm_train(y.flatten(), X, args)
                return cv_accuracy
            self.model = svm_train(y.flatten(), X, args)

        else:
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

            # print(alpha[selector])

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
        if self.backend == Backend.LIBSVM:
            y, _, _ = svm_predict([], X, self.model, "-q")
            return np.array(y).reshape((-1, 1))
        else:
            if self.w is None:
                raise ValueError

            w, b, X_support = self.w, self.b, self.X_support
            K = np.power(np.matmul(X, X_support.T) + 1, self.Q)
            y = np.sum(w * K, axis=1) + b

            y[y > 0] = 1
            y[y <= 0] = -1
            return y.reshape((-1, 1))

    @property
    def num_support_vectors(self):
        if self.backend == Backend.LIBSVM:
            # print(self.model.get_sv_coef())
            return self.model.get_nr_sv()
        else:
            if self.w is None:
                raise ValueError
            return len(self.w)


def error_rate(y1, y2):
    N = y1.shape[0]
    assert y1.shape == y2.shape == (N, 1)
    return np.sum(y1 != y2) / N


def train_and_test(training_X, training_y, test_X, test_y, C, Q, backend, kernel):
    # Solve SVM
    svm = SoftSVM(C=C, Q=Q, kernel=kernel, backend=backend)
    svm.fit(training_X, training_y)

    # Compute in-sample error rate
    hypo_y_in = svm.compute_y(training_X)
    E_in = error_rate(hypo_y_in, training_y)

    # Compute out-sample error rate
    hypo_y_out = svm.compute_y(test_X)
    E_out = error_rate(hypo_y_out, test_y)

    # Get number of support vectors
    num_svs = svm.num_support_vectors

    print("E_in: {:.4f}\nE_out: {:.4f}\nnumber of support vectors: {}".format(E_in, E_out, num_svs))


def digit_vs_all(training_X, training_y, test_X, test_y, digit, C, Q, backend, training_K=None):
    # Construct the training y matrix for one digit vs all others
    target_y = np.zeros_like(training_y)
    target_y[training_y == digit] = 1
    target_y[training_y != digit] = -1

    # Construct the test y matrix for one digit vs all others
    target_test_y = np.zeros_like(test_y)
    target_test_y[test_y == digit] = 1
    target_test_y[test_y != digit] = -1

    train_and_test(training_X, target_y, test_X, target_test_y, C, Q, backend)


def construct_digit_vs_digit(X, y, digit_a, digit_b):
    # Construct y
    a_or_b_indices = (y == digit_a) | (y == digit_b)
    target_y = y[a_or_b_indices].reshape((-1, 1))
    copy_target_y = np.copy(target_y)
    target_y[copy_target_y == digit_a] = 1
    target_y[copy_target_y == digit_b] = -1

    # Construct x
    target_X = X[a_or_b_indices.flatten()]

    return target_X, target_y


def digit_vs_digit(training_X, training_y, test_X, test_y, digit_a, digit_b, C, Q, backend, kernel=Kernel.POLYNOMIAL):
    # Construct the training X and y for one digit vs another
    target_X, target_y = construct_digit_vs_digit(training_X, training_y, digit_a, digit_b)

    # Construct the test X and y for one digit vs another
    target_test_X, target_test_y = construct_digit_vs_digit(test_X, test_y, digit_a, digit_b)

    train_and_test(target_X, target_y, target_test_X, target_test_y, C, Q, backend, kernel=kernel)


def digit_vs_digit_cv(training_X, training_y, digit_a, digit_b, C_values, Q, backend, N):
    # Construct the training X and y for one digit vs another
    target_X, target_y = construct_digit_vs_digit(training_X, training_y, digit_a, digit_b)

    selection_counts = [0] * len(C_values)
    for _ in range(N):
        best_idx = None
        best_accuracy = None
        for idx, C in enumerate(C_values):
            svm = SoftSVM(C=C, Q=Q, kernel=Kernel.POLYNOMIAL, backend=backend)
            accuracy = svm.fit(target_X, target_y, cross_validation=True)
            if best_idx is None or accuracy > best_accuracy:
                best_idx = idx
                best_accuracy = accuracy
        selection_counts[best_idx] += 1
    most_selected_C = C_values[np.argmax(selection_counts)]
    print("Selection counts: {}".format(selection_counts))
    print("Most selected C for {} runs: {}".format(N, most_selected_C))


if __name__ == "__main__":
    # Load training set
    training_set = np.loadtxt("features.train")
    training_y = training_set[:, :1].astype(int)
    training_X = training_set[:, 1:]

    # Load test set
    test_set = np.loadtxt("features.test")
    test_y = test_set[:, :1].astype(int)
    test_X = test_set[:, 1:]

    experiment = 5

    if experiment == 1:
        # Parameters
        target_digits = [1, 3, 5, 7, 9]
        C = 0.01
        Q = 2

        # Pre-compute kernel matrix to save time
        # training_K = np.power(np.matmul(training_X, training_X.T) + 1, Q)

        for digit in tqdm(target_digits):
            # print("digit {} with backend {}:".format(digit, Backend.QPSOLVER))
            # digit_vs_all(training_X, training_y, test_X, test_y, digit, C, Q, backend=Backend.QPSOLVER, training_K=training_K)
            # print("==============")
            print("digit {} with backend {}:".format(digit, Backend.LIBSVM))
            digit_vs_all(training_X, training_y, test_X, test_y, digit, C, Q, backend=Backend.LIBSVM)
            print("################################")
    elif experiment == 2:
        C_values = [0.001, 0.01, 0.1, 1]
        Q = 2
        digit_a = 1
        digit_b = 5
        for C in tqdm(C_values):
            print("digit {} vs digit {}, C: {}:".format(digit_a, digit_b, C))
            digit_vs_digit(training_X, training_y, test_X, test_y, digit_a, digit_b, C, Q, backend=Backend.LIBSVM)
            print("################################")
    elif experiment == 3:
        Q_values = [2, 5]
        C_values = [0.0001, 0.001, 0.01, 1]
        digit_a = 1
        digit_b = 5
        for C in C_values:
            for Q in Q_values:
                print("digit {} vs digit {}, C: {}, Q: {}".format(digit_a, digit_b, C, Q))
                digit_vs_digit(training_X, training_y, test_X, test_y, digit_a, digit_b, C, Q, backend=Backend.LIBSVM)
                print("################################")
    elif experiment == 4:
        digit_a = 1
        digit_b = 5
        N = 100
        Q = 2
        C_values = [0.0001, 0.001, 0.01, 0.1, 1]
        print("digit {} vs digit {}, C_values: {}, Q: {}".format(digit_a, digit_b, C_values, Q))
        digit_vs_digit_cv(training_X, training_y, digit_a, digit_b, C_values, Q, backend=Backend.LIBSVM, N=N)
        print("#####################################")
    elif experiment == 5:
        digit_a = 1
        digit_b = 5
        C_values = [0.01, 1, 100, int(1e4), int(1e6)]
        for C in C_values:
            print("digit {} vs digit {}, C: {}".format(digit_a, digit_b, C))
            digit_vs_digit(training_X, training_y, test_X, test_y, digit_a, digit_b, C, Q=None, backend=Backend.LIBSVM, kernel=Kernel.RBF)
            print("#################################")
