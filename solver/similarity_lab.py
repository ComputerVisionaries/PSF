import numpy as np

from skimage.color import rgb2lab
from skimage.io import imread


def lab_similarity(img1, img2, orientations=['LR', 'RL', 'TD', 'DT']):
    lab_img1 = rgb2lab(img1)
    lab_img2 = rgb2lab(img2)

    rows, cols, _ = lab_img1.shape
    color_difference = None

    similarities = {}

    for orientation in orientations:
        if orientation == 'LR':
            color_difference = lab_img1[:rows, -1] - lab_img2[:rows, 0]
        elif orientation == 'RL':
            color_difference = lab_img2[:rows, -1] - lab_img1[:rows, 0]
        elif orientation == 'TD':
            color_difference = lab_img1[-1, :cols] - lab_img2[0, :cols]
        elif orientation == 'DT':
            color_difference = lab_img2[-1, :cols] - lab_img1[0, :cols]

        similarities[orientation] = np.sqrt(np.sum(np.square(color_difference)))
    
    return similarities


if __name__ == '__main__':
    img0_0 = imread('../images/frog0-0.jpeg')
    img0_1 = imread('../images/frog0-1.jpeg')

    similarities = lab_similarity(img0_0, img0_1)
    print(similarities)
