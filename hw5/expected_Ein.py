# Compute expected in-sample error for a noisy linear target
import numpy as np


def compute_expected_Ein(sigma, d, N):
    return sigma*sigma*(1 - (d+1)/N)


if __name__ == "__main__":
    sigma = 0.1
    d = 8
    Ns = [10, 25, 100, 500, 1000]

    for N in Ns:
        Ein = compute_expected_Ein(sigma, d, N)
        print("for N of {}, get Ein = {}".format(N, Ein))
