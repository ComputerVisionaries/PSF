from enum import Enum
import matplotlib.pyplot as plt
import numpy as np


class RelativePosition(Enum):
    left_right = 0
    right_left = 1
    above_below = 2
    below_above = 3


def rgb_similarity(img1, img2, pos):
    img1_num_rows, img1_num_cols, _ = img1.shape
    img2_num_rows, img2_num_cols, _ = img2.shape

    if pos == RelativePosition.left_right:
            img1_col_index = img1_num_cols - 1
            img2_col_index = 0
            last_row_index = min(img1_num_rows, img2_num_rows)
            img1_col = img1[:last_row_index, img1_col_index, :]
            img2_col = img2[:last_row_index, img2_col_index, :]
            total_error = np.sum(np.square(
                img1_col.astype(np.float64) - img2_col.astype(np.float64)))
            return total_error / last_row_index

    elif pos == RelativePosition.right_left:
            img1_col_index = 0
            img2_col_index = img2_num_cols - 1
            last_row_index = min(img1_num_rows, img2_num_rows)
            img1_col = img1[:last_row_index, img1_col_index, :]
            img2_col = img2[:last_row_index, img2_col_index, :]
            total_error = np.sum(np.square(
                img1_col.astype(np.float64) - img2_col.astype(np.float64)))
            return total_error / last_row_index

    elif pos == RelativePosition.above_below:
        img1_row_index = img1_num_rows - 1
        img2_row_index = 0
        last_col_index = min(img1_num_cols, img2_num_cols)
        img1_row = img1[img1_row_index, :last_col_index, :]
        img2_row = img2[img2_row_index, :last_col_index, :]
        total_error = np.sum(np.square(
            img1_row.astype(np.float64) - img2_row.astype(np.float64)))
        return total_error / last_col_index

    elif pos == RelativePosition.below_above:
        img1_row_index = 0
        img2_row_index = img2_num_rows - 1
        last_col_index = min(img1_num_cols, img2_num_cols)
        img1_row = img1[img1_row_index, :last_col_index, :]
        img2_row = img2[img2_row_index, :last_col_index, :]
        total_error = np.sum(np.square(
            img1_row.astype(np.float64) - img2_row.astype(np.float64)))
        return total_error / last_col_index

    return -1


if __name__ == '__main__':
    img0_0 = plt.imread("frog0-0.jpeg")
    img0_1 = plt.imread("frog0-1.jpeg")

    error = rgb_similarity(img0_0, img0_1, RelativePosition.left_right)
    print(error)

    error = rgb_similarity(img0_0, img0_1, RelativePosition.right_left)
    print(error)

    error = rgb_similarity(img0_0, img0_1, RelativePosition.above_below)
    print(error)

    error = rgb_similarity(img0_0, img0_1, RelativePosition.below_above)
    print(error)
