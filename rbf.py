import numpy as np
from qpsolvers import solve_qp
from tqdm import tqdm
import matplotlib.pyplot as plt
from hw8.libsvm.python.libsvm.svmutil import *


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


class RegularRBF:
    def __init__(self, K, gamma):
        self.K = K
        self.gamma = gamma

        self.centers = None
        self.w = None

        self.rng = np.random.default_rng()

    def k_means(self, X):
        """ Compute K centers in the dataset
        :param X: shape (N, d)
        :return: shape (K, d)
        """
        N, d = X.shape

        centers = self.rng.choice(X, size=self.K, replace=False, shuffle=False)
        assert centers.shape == (self.K, d)

        while True:
            sets = [[] for _ in range(self.K)]
            for x in X:
                best_center_idx = None
                lowest_dist = None
                for idx, center in enumerate(centers):
                    dist = np.linalg.norm(x-center)
                    if best_center_idx is None:
                        lowest_dist = dist
                        best_center_idx = idx
                    elif dist < lowest_dist:
                        lowest_dist = dist
                        best_center_idx = idx
                sets[best_center_idx].append(x)
            new_centers = np.array([np.mean(s, axis=0) for s in sets])
            if np.array_equal(new_centers, centers):
                return centers
            centers = new_centers

    def fit(self, X, y):
        centers = self.k_means(X)

        N = X.shape[0]

        Z = np.ones((N, self.K+1))
        for i, x in enumerate(X):
            for j, mu in enumerate(centers):
                x_diff = x - mu
                Z[i, j+1] = np.exp(-self.gamma * np.inner(x_diff, x_diff))

        w = np.linalg.lstsq(Z, y, rcond=None)[0]

        self.centers = centers
        self.w = w

    def compute_y(self, X):
        N = X.shape[0]

        Z = np.ones((N, self.K+1))
        for i, x in enumerate(X):
            for j, mu in enumerate(self.centers):
                x_diff = x - mu
                Z[i, j+1] = np.exp(-self.gamma * np.inner(x_diff, x_diff))

        y = np.matmul(Z, self.w.reshape((-1, 1)))
        y[y > 0] = 1
        y[y <= 0] = -1
        return y


def error_rate(y1, y2):
    N = y1.shape[0]
    assert y1.shape == y2.shape == (N, 1)
    return np.sum(y1 != y2) / N * 100


if __name__ == "__main__":
    experiment = 17
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
    elif experiment == 14 or experiment == 15:
        K = 12
        gamma = 1.5
        N_exp = 200
        N = 100
        svm_win = []
        for _ in tqdm(range(N_exp)):
            X, y = Target().generate(N)
            X_test, y_test = Target().generate(1000)

            rbf = RegularRBF(K=K, gamma=gamma)
            rbf.fit(X, y)
            y_hat_rbf = rbf.compute_y(X_test)
            E_out_rbf = error_rate(y_hat_rbf, y_test)

            svm = RadialBasisSVM(gamma=gamma, libsvm=True)
            svm.fit(X, y)
            y_hat_svm = svm.compute_y(X_test)
            E_out_svm = error_rate(y_hat_svm, y_test)

            svm_win.append(E_out_svm < E_out_rbf)

        print("Percentage of time that SVM wins: {}%".format(np.sum(svm_win) / N_exp * 100))
    elif experiment == 16:
        gamma = 1.5
        a_counts = 0
        b_counts = 0
        c_counts = 0
        d_counts = 0
        e_counts = 0
        N_exp = 200
        N = 100
        for _ in tqdm(range(N_exp)):
            X, y = Target().generate(N)
            X_test, y_test = Target().generate(1000)

            # K = 9
            rbf = RegularRBF(K=9, gamma=gamma)
            rbf.fit(X, y)

            y_hat = rbf.compute_y(X)
            E_in_9 = error_rate(y_hat, y)

            y_hat = rbf.compute_y(X_test)
            E_out_9 = error_rate(y_hat, y_test)

            # K = 12
            rbf = RegularRBF(K=12, gamma=gamma)
            rbf.fit(X, y)

            y_hat = rbf.compute_y(X)
            E_in_12 = error_rate(y_hat, y)

            y_hat = rbf.compute_y(X_test)
            E_out_12 = error_rate(y_hat, y_test)

            # Compute results
            if E_in_12 < E_in_9 and E_out_12 > E_out_9:
                a_counts += 1
            elif E_in_12 > E_in_9 and E_out_12 < E_out_9:
                b_counts += 1
            elif E_in_12 > E_in_9 and E_out_12 > E_out_9:
                c_counts += 1
            elif E_in_12 < E_in_9 and E_out_12 < E_out_9:
                d_counts += 1
            elif E_in_12 == E_in_9 and E_out_12 == E_out_9:
                e_counts += 1
        print("a: {}, b: {}, c: {}, d: {}, e: {}".format(a_counts, b_counts, c_counts, d_counts, e_counts))
    elif experiment == 17:
        K = 9
        a_counts = 0
        b_counts = 0
        c_counts = 0
        d_counts = 0
        e_counts = 0
        N_exp = 200
        N = 100
        for _ in tqdm(range(N_exp)):
            X, y = Target().generate(N)
            X_test, y_test = Target().generate(1000)

            # gamma = 1.5
            rbf = RegularRBF(K=K, gamma=1.5)
            rbf.fit(X, y)

            y_hat = rbf.compute_y(X)
            E_in_9 = error_rate(y_hat, y)

            y_hat = rbf.compute_y(X_test)
            E_out_9 = error_rate(y_hat, y_test)

            # gamma = 2
            rbf = RegularRBF(K=K, gamma=2)
            rbf.fit(X, y)

            y_hat = rbf.compute_y(X)
            E_in_12 = error_rate(y_hat, y)

            y_hat = rbf.compute_y(X_test)
            E_out_12 = error_rate(y_hat, y_test)

            # Compute results
            if E_in_12 < E_in_9 and E_out_12 > E_out_9:
                a_counts += 1
            elif E_in_12 > E_in_9 and E_out_12 < E_out_9:
                b_counts += 1
            elif E_in_12 > E_in_9 and E_out_12 > E_out_9:
                c_counts += 1
            elif E_in_12 < E_in_9 and E_out_12 < E_out_9:
                d_counts += 1
            elif E_in_12 == E_in_9 and E_out_12 == E_out_9:
                e_counts += 1
        print("a: {}, b: {}, c: {}, d: {}, e: {}".format(a_counts, b_counts, c_counts, d_counts, e_counts))