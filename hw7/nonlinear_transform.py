# Linear regression on a nonlinear transformed feature space
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

    def solve(self, Z, y):
        assert y.shape[1] == 1
        # w = (Z^T Z)^(-1)Z^T y
        w = np.matmul(np.linalg.pinv(Z), y)
        return w


def calculate_error(Z, w, y):
    y_hat = (np.matmul(Z, w) > 0).astype(int)
    y_hat[y_hat != 1] = -1
    # Compute error
    N = Z.shape[0]
    error = np.sum(y_hat != y) / N
    return error


if __name__ == "__main__":
    insample_data = np.loadtxt("in.dta")
    outsample_data = np.loadtxt("out.dta")
    training_data = insample_data[:25]
    validation_data = insample_data[25:]
    assert len(validation_data) == 10

    training_data, validation_data = np.copy(validation_data), np.copy(training_data)
    assert len(training_data) == 10 and len(validation_data) == 25

    X = training_data[:, :2]
    y = training_data[:, 2:]
    Z = nonlinear_transform(X)

    # Validation data
    validation_y = validation_data[:, 2:].astype(int)
    validation_X = validation_data[:, :2]
    validation_Z = nonlinear_transform(validation_X)

    # Test data
    test_y = outsample_data[:, 2:].astype(int)
    test_X = outsample_data[:, :2]
    test_Z = nonlinear_transform(test_X)

    for k in range(3, 8):
        features = Z[:, :k+1]
        w = LinearRegression().solve(features, y)

        # Evaludate validation error
        validation_features = validation_Z[:, :k+1]
        validation_error = calculate_error(validation_features, w, validation_y)

        # Evaluate test error
        test_features = test_Z[:, :k+1]
        test_error = calculate_error(test_features, w, test_y)

        print("k = {}, validation error: {}, test error: {}".format(k, validation_error, test_error))
