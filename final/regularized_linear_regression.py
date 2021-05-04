import numpy as np


class LinearRegressionWeightDecay:
    def __init__(self, lambda_, nonlinear_transform):
        """
        :param lambda_: regularization param
        :param nonlinear_transform: function
        """
        self.lambda_ = lambda_
        self.nonlinear_transform = nonlinear_transform
        self.w = None

    def transform(self, X):
        if self.nonlinear_transform is not None:
            return self.nonlinear_transform(X)
        return np.copy(X)

    def fit(self, X, y):
        """
        :param X: shape (N, m)
        :param y: shape (N, 1)
        :return: weight vector of shape (num_weights, 1)
        """
        Z = self.transform(X)
        num_w = Z.shape[1]
        assert y.shape[1] == 1 and y.shape[0] == Z.shape[0]
        # w = (Z^T Z + lambda*I)^(-1)Z^T y
        A = np.matmul(Z.T, Z) + self.lambda_ * np.identity(num_w)
        B = np.matmul(Z.T, y)
        w = np.matmul(np.linalg.inv(A), B)

        self.w = w

    def compute_y(self, X):
        Z = self.transform(X)
        y_hat = (np.matmul(Z, self.w) > 0).astype(int)
        y_hat[y_hat != 1] = -1
        return y_hat


def error_rate(y1, y2):
    N = y1.shape[0]
    assert y1.shape == y2.shape == (N, 1)
    return np.sum(y1 != y2) / N * 100


def train_and_test(training_X, training_y, test_X, test_y, lambda_, nonlinear_transform):
    # Solve regularized linear regression
    reg_linear_regression = LinearRegressionWeightDecay(
        lambda_=lambda_,
        nonlinear_transform=nonlinear_transform
    )
    reg_linear_regression.fit(training_X, training_y)

    # Compute in-sample error rate
    hypo_y_in = reg_linear_regression.compute_y(training_X)
    E_in = error_rate(hypo_y_in, training_y)

    # Compute out-sample error rate
    hypo_y_out = reg_linear_regression.compute_y(test_X)
    E_out = error_rate(hypo_y_out, test_y)

    print("E_in: {:.3f}%\nE_out: {:.3f}%".format(E_in, E_out))


def digit_vs_all(training_X, training_y, test_X, test_y, digit, lambda_, nonlinear_transform):
    # Construct the training y matrix for one digit vs all others
    target_y = np.zeros_like(training_y)
    target_y[training_y == digit] = 1
    target_y[training_y != digit] = -1

    # Construct the test y matrix for one digit vs all others
    target_test_y = np.zeros_like(test_y)
    target_test_y[test_y == digit] = 1
    target_test_y[test_y != digit] = -1

    train_and_test(training_X, target_y, test_X, target_test_y, lambda_, nonlinear_transform)


def add_bias_transform(X):
    N = X.shape[0]
    Z = np.ones([N, 3])
    Z[:, 1:] = X[:, :]
    return Z


def nonlinear_feature_transform(X):
    N = X.shape[0]
    Z = np.ones([N, 6])
    Z[:, 1:3] = X[:, :]
    x1 = X[:, 0]
    x2 = X[:, 1]
    Z[:, 3] = x1*x2
    Z[:, 4] = x1*x1
    Z[:, 5] = x2*x2
    return Z


if __name__ == "__main__":
    # Load training set
    training_set = np.loadtxt("../hw8/features.train")
    training_y = training_set[:, :1].astype(int)
    training_X = training_set[:, 1:]

    # Load test set
    test_set = np.loadtxt("../hw8/features.test")
    test_y = test_set[:, :1].astype(int)
    test_X = test_set[:, 1:]

    experiment = 2

    if experiment == 1:
        lambda_ = 1
        for digit in range(5, 10):
            print("digit: {}, lambda: {}, no nonlinear transform".format(digit, lambda_))
            digit_vs_all(training_X, training_y, test_X, test_y, digit, lambda_, add_bias_transform)
            print("===========")
    elif experiment == 2:
        lambda_ = 1
        for digit in range(0, 5):
            print("digit: {}, lambda: {}, with nonlinear transform".format(digit, lambda_))
            digit_vs_all(training_X, training_y, test_X, test_y, digit, lambda_, nonlinear_feature_transform)
            print("===========")
