# Compute expected value of min(e1, e2) to demonstrate validation bias
import numpy as np


if __name__ == "__main__":
    n_trials = 100000
    e1 = np.random.uniform(size=n_trials)
    e2 = np.random.uniform(size=n_trials)
    min_e = np.minimum(e1, e2)
    assert(len(min_e) == n_trials)

    expected_min_e = np.mean(min_e)
    print("expected min e: {}".format(expected_min_e))
