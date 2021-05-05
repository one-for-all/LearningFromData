import numpy as np
from qpsolvers import solve_qp


class SVM:
    def __init__(self):
        pass

    def num_svs(self, X, y):
        N, d = X.shape

        # Gram matrix
        K = np.power(np.matmul(X, X.T) + 1, 2)

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

        # Solve the QP program
        alpha = solve_qp(
            P=P, q=q, A=A, b=b, lb=lb, solver="cvxopt"
        )

        # Compute number of support vectors
        selector = ~np.isclose(alpha, 0, atol=1e-4)
        return sum(selector)


if __name__ == "__main__":
    dataset = np.array([[1, 0, -1],
                        [0, 1, -1],
                        [0, -1, -1],
                        [-1, 0, 1],
                        [0, 2, 1],
                        [0, -2, 1],
                        [-2, 0, 1]])
    X = dataset[:, :2]
    y = dataset[:, 2:]

    num_svs = SVM().num_svs(X, y)
    print("number of support vectors: {}".format(num_svs))
