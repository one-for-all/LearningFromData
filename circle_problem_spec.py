# Circularly Separable Problem specification with noise
import numpy as np


class Problem:
    def __init__(self):
        self.x1_range = (-1, 1)
        self.x2_range = (-1, 1)

    def sample(self, num_points):
        x1s = np.random.uniform(self.x1_range[0], self.x1_range[1], num_points)
        x2s = np.random.uniform(self.x2_range[0], self.x2_range[1], num_points)
        ys = (x1s**2 + x2s**2 - 0.6 > 0).astype(np.int)
        ys[ys == 0] = -1

        # Flip 10% labels for noise
        N = len(ys)
        random_indices = np.random.randint(N, size=N//10)
        ys[random_indices] *= -1
        return x1s, x2s, ys
