import numpy as np
import matplotlib.pyplot as plt

from RelativePosition import RelativePosition


def similarity(piece1, piece2, pos):
    # Edge keys: r, b, l, t
    if pos == 'LR':
        points1 = piece1.get_side('r')
        points2 = piece2.get_side('l')

        total_error = np.sum(np.square(
            points1.astype(np.float64) - points2.astype(np.float64)))
        return np.sqrt(total_error)

    elif pos == 'TD':
        points1 = piece1.getSide('t')
        points2 = piece2.getSide('b')

        total_error = np.sum(np.square(
            points1.astype(np.float64) - points2.astype(np.float64)))
        return np.sqrt(total_error)

    return -1
