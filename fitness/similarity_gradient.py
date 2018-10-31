import matplotlib.pyplot as plt
import numpy as np
from RelativePosition import RelativePosition


def Gil(img):
    num_rows, num_cols, num_clrs = img.shape
    gradients = np.zeros((num_rows, num_clrs))

    for p in range(num_rows):
        for c in range(num_clrs):
            gradients[p, c] = img[p, num_cols-1, c].astype(np.float64)
            - img[p, num_cols-2, c].astype(np.float64)

    return gradients


def Gir(img):
    num_rows, num_cols, num_clrs = img.shape
    gradients = np.zeros((num_rows, num_clrs))

    for p in range(num_rows):
        for c in range(num_clrs):
            gradients[p, c] = img[p, 0, c].astype(np.float64)
            - img[p, 1, c].astype(np.float64)
    return gradients


def Ui(gi, c):
    p, _ = gi.shape
    return np.sum(gi.astype(np.float64)[:, c]) / p


def Si(gi):
    return np.cov(gi.T)


def Gijlr(img_i, img_j):
    num_rows, num_cols, num_clrs = img_i.shape
    gradients = np.zeros((num_rows, num_clrs))

    for p in range(num_rows):
        for c in range(num_clrs):
            gradients[p, c] = img_i[p, 0, c].astype(np.float64)
            - img_j[p, num_cols-1, c].astype(np.float64)

    return gradients


def Gjirl(img_i, img_j):
    num_rows, num_cols, num_clrs = img_i.shape
    gradients = np.zeros((num_rows, num_clrs))

    for p in range(num_rows):
        for c in range(num_clrs):
            gradients[p, c] = img_i[p, num_cols-1, c].astype(np.float64)
            - img_j[p, 0, c].astype(np.float64)

    return gradients


def Eq3(mat1, mat2):
    term = 0
    term += mat1[0] * (mat1[0] * mat2[0, 0]
                       + mat1[1] * mat2[0, 1] + mat1[2] * mat2[0, 2])
    term += mat1[1] * (mat1[0] * mat2[1, 0]
                       + mat1[1] * mat2[1, 1] + mat1[2] * mat2[1, 2])
    term += mat1[2] * (mat1[0] * mat2[2, 0]
                       + mat1[1] * mat2[2, 1] + mat1[2] * mat2[2, 2])
    return term


def mahalanobis_gradient_compat(img1, img2):
    num_rows, _, _ = img1.shape

    gil = Gil(img1)
    uil = np.array([Ui(gil, 0), Ui(gil, 1), Ui(gil, 2)])
    sil = Si(gil)
    gijlr = Gijlr(img1, img2)

    dlr = 0
    for p in range(num_rows):
        mat1 = gijlr[p, :] - uil
        mat2 = np.linalg.inv(sil)
        dlr += Eq3(mat1, mat2)

    gir = Gir(img2)
    uir = np.array([Ui(gir, 0), Ui(gir, 1), Ui(gir, 2)])
    sir = Si(gir)
    gjirl = Gjirl(img1, img2)

    drl = 0
    for p in range(num_rows):
        mat1 = gjirl[p, :] - uir
        mat2 = np.linalg.inv(sir)
        drl += Eq3(mat1, mat2)

    return dlr + drl


def gradient_similarity(img1, img2, pos):
    if pos == RelativePosition.LEFT_RIGHT:
        return mahalanobis_gradient_compat(img1, img2)
    elif pos == RelativePosition.RIGHT_LEFT:
        return mahalanobis_gradient_compat(img2, img1)
    elif pos == RelativePosition.ABOVE_BELOW:
        img1 = np.transpose(img1, [1, 0, 2])
        img2 = np.transpose(img2, [1, 0, 2])
        return mahalanobis_gradient_compat(img2, img1)
    elif pos == RelativePosition.BELOW_ABOVE:
        img1 = np.transpose(img1, [1, 0, 2])
        img2 = np.transpose(img2, [1, 0, 2])
        return mahalanobis_gradient_compat(img1, img2)


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
