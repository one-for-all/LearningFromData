# Perform gradient descent on a provided error function
import numpy as np


def evaluate_E(x):
    u, v = x
    return (u*np.exp(v) - 2*v*np.exp(-u))**2


def evaluate_dEdu(x):
    u, v = x
    dEdu = 2*(np.exp(v) + 2*v*np.exp(-u))*(u*np.exp(v) - 2*v*np.exp(-u))
    return dEdu


def evaluate_dEdv(x):
    u, v = x
    dEdv = 2*(u*np.exp(v) - 2*v*np.exp(-u))*(u*np.exp(v) - 2*np.exp(-u))
    return dEdv


if __name__ == "__main__":
    eta = 0.1
    x = np.array([1.0, 1.0])
    iters = 0
    while iters < 15:
        dEdu = evaluate_dEdu(x)
        x -= eta*np.array([dEdu, 0])
        dEdv = evaluate_dEdv(x)
        x -= eta*np.array([0, dEdv])
        iters += 1

    E = evaluate_E(x)
    print("x = {} at iteration {}, with E = {}".format(x, iters, E))
