# Utilities
import numpy as np


def draw_line(plt, w, color='g'):
    x1s = np.linspace(-1, 1, 10)
    x2s = (-w[0] - w[1] * x1s) / w[2]
    plt.plot(x1s, x2s, color=color)


def draw_points(plt, x1s, x2s, ys):
    for x1, x2, y in zip(x1s, x2s, ys):
        plt.scatter(x1, x2, color=('b' if y == 1 else 'r'))
