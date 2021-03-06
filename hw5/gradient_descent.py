# Perform gradient descent on a provided error function
import numpy as np


def evaluate_E(x):
    u, v = x
    return (u*np.exp(v) - 2*v*np.exp(-u))**2


def evaluate_gradient(x):
    u, v = x
    dEdu = 2*(np.exp(v) + 2*v*np.exp(-u))*(u*np.exp(v) - 2*v*np.exp(-u))
    dEdv = 2*(u*np.exp(v) - 2*v*np.exp(-u))*(u*np.exp(v) - 2*np.exp(-u))
    return np.array([dEdu, dEdv])


if __name__ == "__main__":
    eta = 0.1
    x = np.array([1.0, 1.0])
    iters = 0
    while True:
        E = evaluate_E(x)
        if E < 10e-14:
            break
        gradient = evaluate_gradient(x)
        x -= eta*gradient
        iters += 1

    print("x = {} at iteration {}, with E = {}".format(x, iters, E))
