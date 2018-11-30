import numpy as np
import matplotlib.pyplot as plt

from RelativePosition import RelativePosition


def similarity(piece1, piece2, pos):
    # Edge keys: r, b, l, t
    if pos == 'LR':
        side1 = piece1.get_side('r')['shape']
        side2 = piece2.get_side('l')['shape']

    elif pos == 'TD':
        side1 = piece1.get_side('b')['shape']
        side2 = piece2.get_side('t')['shape']
    # if side1 == 0 or side2 == 0
    if side1 * -1 == side2 and side1 != 0:
        return 0
    else:
        return 1

def similarity_rgb(piece1, piece2, pos):
    # Edge keys: r, b, l, t
    if pos == 'LR':
        side1 = piece1.get_side('r')
        side2 = piece2.get_side('l')
        points1 = [a for a in side1['edge'] if list(a) != [0, 0, 0]]
        points2 = [a for a in side2['edge'] if list(a) != [0, 0, 0]]
        lim = min(len(points1), len(points2))
        points1, points2 = np.asarray(points1[:lim]), np.asarray(points2[:lim])
        # import pdb; pdb.set_trace()
        total_error = np.sum(np.square(
            points1.astype(np.float64) - points2.astype(np.float64)))
        return np.sqrt(total_error)

    elif pos == 'TD':
        side1 = piece1.get_side('b')
        side2 = piece2.get_side('t')
        points1 = [a for a in side1['edge'] if list(a) != [0, 0, 0]]
        points2 = [a for a in side2['edge'] if list(a) != [0, 0, 0]]
        lim = min(len(points1), len(points2))
        points1, points2 = np.asarray(points1[:lim]), np.asarray(points2[:lim])

        total_error = np.sum(np.square(
            points1.astype(np.float64) - points2.astype(np.float64)))
        return np.sqrt(total_error)

    return -1
