import numpy as np
import matplotlib.pyplot as plt

from RelativePosition import RelativePosition


def similarity(piece1, piece2, pos, num_pts=100):
    # Edge keys: r, b, l, t
    # TODO: Check that order of comparison is correct
    # TODO: Compute edge histograms for robustness
    if pos == RelativePosition.LEFT_RIGHT:
        points1 = piece1.getSide('r', num_pts)
        # plt.imshow(piece1.image)
        # rows = [pt[0] for pt in points1]
        # cols = [pt[1] for pt in points1]
        # plt.plot(cols, rows, 'ro')
        # plt.show()
        points2 = piece2.getSide('l', num_pts)

        total_error = np.sum(np.square(
            points1.astype(np.float64) - points2.astype(np.float64)))
        return np.sqrt(total_error) / num_pts

    elif pos == RelativePosition.RIGHT_LEFT:
        points1 = piece1.getSide('l', num_pts)
        points2 = piece2.getSide('r', num_pts)

        total_error = np.sum(np.square(
            points1.astype(np.float64) - points2.astype(np.float64)))
        return np.sqrt(total_error) / num_pts

    elif pos == RelativePosition.ABOVE_BELOW:
        points1 = piece1.getSide('t', num_pts)
        points2 = piece2.getSide('b', num_pts)

        total_error = np.sum(np.square(
            points1.astype(np.float64) - points2.astype(np.float64)))
        return np.sqrt(total_error) / num_pts

    elif pos == RelativePosition.BELOW_ABOVE:
        points1 = piece1.getSide('b', num_pts)
        points2 = piece2.getSide('t', num_pts)

        total_error = np.sum(np.square(
            points1.astype(np.float64) - points2.astype(np.float64)))
        return np.sqrt(total_error) / num_pts

    return -1
