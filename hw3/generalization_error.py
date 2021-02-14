import numpy as np


def compute_generalization_prob(epsilon, M, N):
    return 2 * M * np.exp(-2 * epsilon**2 * N)


if __name__ == "__main__":
    epsilon = 0.05
    M = 100
    found = False
    for N in [500, 1000, 1500, 2000]:
        if compute_generalization_prob(epsilon, M, N) <= 0.03:
            print("least number of N required: {}".format(N))
            found = True
            break
    if not found:
        print("more examples are required")