# Compute the expected linear model for learning the sin(pi x) function
import numpy as np


def sample_points():
    x = np.random.uniform(-1, 1, 2)
    y = np.sin(np.pi * x)
    return x, y


def compute_linear(x, y):
    X = np.array(x).reshape((-1, 1))
    a = np.matmul(np.linalg.pinv(X), y)
    return a


if __name__ == "__main__":
    a_values = []
    for _ in range(20000):
        x, y = sample_points()
        a = compute_linear(x, y)
        a_values.append(a)

    avg_a = np.mean(a_values)
    print("average value of a: {}".format(avg_a))

    var_values = []
    for a in a_values:
        var = (a - avg_a)**2/3
        var_values.append(var)

    print("expected var: {}".format(np.mean(var_values)))
