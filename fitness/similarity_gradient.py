import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import sobel
from RelativePosition import RelativePosition


def gradient_similarity(img1, img2, pos):
    img1_num_rows, img1_num_cols, _ = img1.shape
    img2_num_rows, img2_num_cols, _ = img2.shape

    if pos == RelativePosition.LEFT_RIGHT:
        hstack = np.hstack((img1, img2))[:, img1_num_cols-3:img1_num_cols+2, :]
        gradients = sobel(hstack, axis=0)
        return np.sum(np.square(gradients.astype(np.float64))) / img1_num_rows

    elif pos == RelativePosition.RIGHT_LEFT:
        hstack = np.hstack((img2, img1))[:, img2_num_cols-3:img2_num_cols+2, :]
        gradients = sobel(hstack, axis=0)
        return np.sum(np.square(gradients.astype(np.float64))) / img2_num_rows

    elif pos == RelativePosition.ABOVE_BELOW:
        vstack = np.vstack((img1, img2))[img1_num_rows-3:img1_num_rows+2, :, :]
        gradients = sobel(vstack, axis=1)
        return np.sum(np.square(gradients.astype(np.float64))) / img1_num_cols

    elif pos == RelativePosition.BELOW_ABOVE:
        vstack = np.vstack((img2, img1))[img2_num_rows-3:img2_num_rows+2, :, :]
        gradients = sobel(vstack, axis=1)
        return np.sum(np.square(gradients.astype(np.float64))) / img2_num_cols

    return -1


if __name__ == '__main__':
    img0_0 = plt.imread("../images/frog0-0.jpeg")
    img0_1 = plt.imread("../images/frog0-1.jpeg")

    error = gradient_similarity(img0_0, img0_1, RelativePosition.LEFT_RIGHT)
    print(error)

    error = gradient_similarity(img0_0, img0_1, RelativePosition.RIGHT_LEFT)
    print(error)

    error = gradient_similarity(img0_0, img0_1, RelativePosition.ABOVE_BELOW)
    print(error)

    error = gradient_similarity(img0_0, img0_1, RelativePosition.BELOW_ABOVE)
    print(error)
