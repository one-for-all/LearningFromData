import numpy as np
from qpsolvers import solve_qp
from tqdm import tqdm
import matplotlib.pyplot as plt
from hw8.libsvm.python.svmutil import *


class Target:
    def __init__(self):
        self.x1_range = [-1, 1]
        self.x2_range = [-1, 1]

    def generate(self, num_points):
        """
        :param num_points: int
        :return: X of shape (N, 2), y of shape (N, 1)
        """
        x1s = np.random.uniform(self.x1_range[0], self.x1_range[1], num_points)
        x2s = np.random.uniform(self.x2_range[0], self.x2_range[1], num_points)
        ys = (x2s - x1s + 0.25*np.sin(np.pi*x1s) > 0).astype(int)
        ys[ys == 0] = -1

        X = np.vstack([x1s, x2s]).T
        y = ys.reshape((-1, 1))
        assert X.shape[1] == 2
        assert y.shape[1] == 1
        assert X.shape[0] == y.shape[0]
        return X, y

    def compute_y(self, x1, x2):
        y = (x2 - x1 + 0.25*np.sin(np.pi*x1) > 0)
        y = 1 if y else -1
        return y

    def plot(self, model=None):
        x1s = []
        x2s = []
        colors = []
        for x1 in np.linspace(self.x1_range[0], self.x1_range[1], 100):
            for x2 in np.linspace(self.x2_range[0], self.x2_range[1], 100):
                if not model:
                    y = self.compute_y(x1, x2)
                else:
                    y = model.compute_y(np.array([[x1, x2]]))[[0]]
                color = 'r' if y == 1 else 'b'

                x1s.append(x1)
                x2s.append(x2)
                colors.append(color)

        plt.scatter(x1s, x2s, c=colors, s=0.5)
        plt.show()


class RadialBasisSVM:
    def __init__(self, gamma, libsvm=False):
        self.gamma = gamma
        self.libsvm = libsvm

        self.w = None
        self.b = None
        self.X_support = None

    def compute_K(self, X1, X2):
        N1, N2 = X1.shape[0], X2.shape[0]
        K = np.zeros((N1, N2))
        for i in range(N1):
            for j in range(N2):
                x_diff = X1[i] - X2[j]
                K[i, j] = np.exp(-self.gamma * np.inner(x_diff, x_diff))
        return K

    def fit(self, X, y):
        if self.libsvm:
            kernel_type = 2
            args = '-t {} -g {} -q -c 10000'.format(kernel_type, self.gamma)
            self.model = svm_train(y.flatten(), X, args)
            return

        N, d = X.shape

        # Gram matrix
        K = self.compute_K(X, X)

        # Quadratic matrix
        Y = np.matmul(y, y.T)
        P = np.multiply(K, Y).astype(float)

        # Linear matrix
        q = -np.ones(N)

        # Equality constraint matrix
        A = y.T.astype(float)

        # Equality target matrix
        b = np.array([[0.0]])

        # Lower-bound on the variables
        lb = np.zeros(N)
        ub = None  # 10000 * np.ones(N)

        # Solve the QP program
        alpha = solve_qp(
            P=P, q=q, A=A, b=b, lb=lb, ub=ub, solver="cvxopt"
        )

        assert alpha is not None

        # Compute final hypothesis w and support vectors
        selector = ~np.isclose(alpha, 0)
        w = alpha[selector] * y.flatten()[selector]

        X_support = X[selector]

        # Compute b
        x_s = X_support[:1, :]
        assert x_s.shape == (1, d)
        y_s = y.flatten()[0]
        K_s = self.compute_K(x_s, X_support)
        b = y_s - np.sum(w * K_s)

        self.w = w
        self.b = b
        self.X_support = X_support

    def compute_y(self, X):
        if self.libsvm:
            y, _, _ = svm_predict([], X, self.model, "-q")
            return np.array(y).reshape((-1, 1))

        if self.w is None:
            raise ValueError

        w, b, X_support = self.w, self.b, self.X_support
        K = self.compute_K(X, X_support)
        y = np.sum(w * K, axis=1) + b

        y[y > 0] = 1
        y[y <= 0] = -1
        return y.reshape((-1, 1))


def error_rate(y1, y2):
    N = y1.shape[0]
    assert y1.shape == y2.shape == (N, 1)
    return np.sum(y1 != y2) / N * 100


if __name__ == "__main__":
    experiment = 13
    if experiment == 13:
        N = 100
        gamma = 1.5
        N_exp = 500
        not_sep_count = 0
        E_ins = []
        for _ in tqdm(range(N_exp)):
            X, y = Target().generate(N)
            svm = RadialBasisSVM(gamma=gamma, libsvm=True)
            svm.fit(X, y)
            y_hat = svm.compute_y(X)

            E_in = error_rate(y_hat, y)
            E_ins.append(E_in)
            if not np.isclose(E_in, 0):
                not_sep_count += 1

        percentage = not_sep_count / N_exp * 100
        print("percentage of time that is not separable: {}%".format(percentage))
        print("mean E_in: {}%".format(np.mean(E_ins)))

