# Apply linear regression with weight decay regularization
import numpy as np


def nonlinear_transform(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    Z = np.array([np.ones_like(x1),
                  x1,
                  x2,
                  x1*x1,
                  x2*x2,
                  x1*x2,
                  np.abs(x1-x2),
                  np.abs(x1+x2)]).T
    assert Z.shape == (X.shape[0], 8)
    return Z


class LinearRegression:
    def __init__(self):
        pass

    def solve(self, X, y):
        Z = nonlinear_transform(X)
        assert y.shape[1] == 1
        # w = (Z^T Z)^(-1)Z^T y
        w = np.matmul(np.linalg.pinv(Z), y)
        return w


class LinearRegressionWeightDecay:
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def solve(self, X, y):
        Z = nonlinear_transform(X)
        num_w = Z.shape[1]
        assert y.shape[1] == 1
        # w = (Z^T Z + lambda*I)^(-1)Z^T y
        A = np.matmul(Z.T, Z) + self.lambda_ * np.identity(num_w)
        B = np.matmul(Z.T, y)
        w = np.matmul(np.linalg.inv(A), B)
        return w


def calculate_error(data_set, w):
    X = data_set[:, :2]
    y = data_set[:, 2:].astype(int)
    Z = nonlinear_transform(X)
    y_hat = (np.matmul(Z, w) > 0).astype(int)
    y_hat[y_hat != 1] = -1
    # Compute error
    N = data_set.shape[0]
    error = np.sum(y_hat != y) / N
    return error


if __name__ == "__main__":
    training_set = np.loadtxt("in.dta")
    test_set = np.loadtxt("out.dta")

    X_train = training_set[:, :2]
    y_train = training_set[:, 2:]

    for k in [2, 1, 0, -1, -2]:
        lambda_ = 10**k
        linear_regression = LinearRegressionWeightDecay(lambda_)
        w = linear_regression.solve(X_train, y_train)

        in_sample_error = calculate_error(training_set, w)
        out_sample_error = calculate_error(test_set, w)

        print("k = {}".format(k))
        print("in-sample error: {}".format(in_sample_error))
        print("out-sample error: {}".format(out_sample_error))
        print("=======")

