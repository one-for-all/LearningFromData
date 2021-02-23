# Compute values of several bounds on the generalization error
import numpy as np
import matplotlib.pyplot as plt


def vc_bound(delta, d_vc, N):
    return np.sqrt(8/N * np.log(4*((2*N)**d_vc + 1)/delta))


def rademacher_penalty_bound(delta, d_vc, N):
    c = float(2*N*(N**d_vc + 1))
    a = np.sqrt(2*np.log(c)/N)
    b = np.sqrt(2/N*np.log(1/delta))
    return a + b + 1/N


def parrondo_vand_den_broek(delta, d_vc, N, initial_guess=1):
    epsilon = initial_guess - 1
    epsilon_next = initial_guess
    while abs(epsilon - epsilon_next) > 1e-4:
        epsilon = epsilon_next
        epsilon_next = np.sqrt(1/N*(2*epsilon + np.log(6*((2*N)**d_vc + 1)/delta)))
    return epsilon


def devroye(delta, d_vc, N, initial_guess=1.0):
    N = float(N)
    epsilon = initial_guess - 1
    epsilon_next = initial_guess
    while abs(epsilon - epsilon_next) > 1e-4:
        epsilon = epsilon_next
        a = np.log(4) + 2*d_vc*np.log(N) - np.log(delta)
        b = 1/(2*N) * (4*epsilon*(1+epsilon) + a)
        epsilon_next = np.sqrt(b)
    return epsilon


if __name__ == "__main__":
    N_values = [5, 10, 100, 1000, 10000]
    delta = 0.05
    d_vc = 50

    vc_bound_values = []
    rp_bound_values = []
    pvdb_values = []
    d_values = []

    for N in N_values:
        vc_bound_values.append(vc_bound(delta, d_vc, N))
        rp_bound_values.append(rademacher_penalty_bound(delta, d_vc, N))
        pvdb_values.append(parrondo_vand_den_broek(delta, d_vc, N))
        d_values.append(devroye(delta, d_vc, N))

    plt.plot(N_values, vc_bound_values, label="VC")
    plt.plot(N_values, rp_bound_values, label="RP")
    plt.plot(N_values, pvdb_values, label="PVDB")
    plt.plot(N_values, d_values, label="Devroye")
    plt.legend()
    plt.xscale("log")
    plt.show()
