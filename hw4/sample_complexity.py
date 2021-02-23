# Solve for sample complexity by iterative method
import numpy as np


def solve_sample_complexity(epsilon, delta, d_vc, initial_guess=10000):
    N = None
    N_next = initial_guess
    while N != N_next:
        N = N_next
        N_next = int( 8/epsilon**2 * np.log(4*((2*N)**d_vc + 1)/ delta) )
    return N


if __name__ == "__main__":
    N = solve_sample_complexity(epsilon=0.05, delta=0.05, d_vc=10, initial_guess=400000)
    print("# samples required: {}".format(N))
