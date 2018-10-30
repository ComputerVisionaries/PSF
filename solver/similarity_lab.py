import numpy as np

from enum import Enum
from skimage.color import rgb2lab
from skimage.io import imread

class RelativePosition(Enum):
    LEFT_RIGHT = 0
    RIGHT_LEFT = 1
    ABOVE_BELOW = 2
    BELOW_ABOVE = 3

def lab_similarity(img1, img2, pos):
    lab_img1 = rgb2lab(img1)
    lab_img2 = rgb2lab(img2)

    rows, cols, _ = lab_img1.shape
    diff = None

    if pos == RelativePosition.LEFT_RIGHT:
        diff = lab_img1[:rows, -1] - lab_img2[:rows, 0]
    elif pos == RelativePosition.RIGHT_LEFT:
        diff = lab_img2[:rows, -1] - lab_img1[:rows, 0]
    elif pos == RelativePosition.ABOVE_BELOW:
        diff = lab_img1[-1, :cols] - lab_img2[0, :cols]
    elif pos == RelativePosition.BELOW_ABOVE:
        diff = lab_img2[-1, :cols] - lab_img1[0, :cols]

    return np.sqrt(np.sum(np.square(diff)))


if __name__ == '__main__':
    img0_0 = imread('../images/frog0-0.jpeg')
    img0_1 = imread('../images/frog0-1.jpeg')

    similarity = lab_similarity(img0_0, img0_1, RelativePosition.LEFT_RIGHT)
    print(similarity)
