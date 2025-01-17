# Perception Learning Algorithm
import numpy as np


class PerceptronLearningAlgorithm:
    def __init__(self):
        pass

    @staticmethod
    def compute_label(w, x1, x2):
        y = 1 if w[0] + w[1]*x1 + w[2]*x2 > 0 else -1
        return y

    @staticmethod
    def compute_multiple_labels(w, x1s, x2s):
        ys = (w[0] + w[1] * x1s + w[2] * x2s > 0).astype(int)
        ys[ys == 0] = -1
        return ys

    def solve(self, x1s, x2s, ys, w_initial=None):
        if w_initial is None:
            w = np.zeros_like(ys)
        else:
            w = np.copy(w_initial)

        n_iters = 0
        while True:
            has_wrong = False
            for x1, x2, target_y in zip(x1s, x2s, ys):
                hypothesis_y = self.compute_label(w, x1, x2)
                if hypothesis_y != target_y:
                    has_wrong = True
                    w += target_y * np.array([1, x1, x2])
                    n_iters += 1
            if not has_wrong:
                break

        return w, n_iters
