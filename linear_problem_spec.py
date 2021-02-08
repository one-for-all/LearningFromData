# Linearly Separable Problem Specification
import numpy as np


class Problem:
    def __init__(self):
        self.x1_range = (-1, 1)
        self.x2_range = (-1, 1)

        # Pick two random points for forming a line
        x1_values = np.random.uniform(self.x1_range[0], self.x1_range[1], 2)
        x2_values = np.random.uniform(self.x2_range[0], self.x2_range[1], 2)
        p1 = np.array([x1_values[0], x2_values[0]])
        p2 = np.array([x1_values[1], x2_values[1]])
        delta_p = p1 - p2

        # Solve for the parameters of the separating line
        # w0 + w1*x1 + w2*x2 = 0
        w0 = 1  # Fix at 1
        w2 = 1/(p1[0]*delta_p[1]/delta_p[0] - p1[1])
        w1 = -w2 * delta_p[1]/delta_p[0]

        self.w = np.array((w0, w1, w2))

    def sample(self, num_points):
        x1s = np.random.uniform(self.x1_range[0], self.x1_range[1], num_points)
        x2s = np.random.uniform(self.x2_range[0], self.x2_range[1], num_points)
        ys = (self.w[0] + self.w[1]*x1s + self.w[2]*x2s > 0).astype(np.int)
        ys[ys == 0] = -1
        return x1s, x2s, ys
