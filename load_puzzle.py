import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imread, imshow, imsave
from skimage.feature import corner_peaks


from puzzle.physicalPiece import PhysicalPiece as piece
from puzzle.physicalPiece import Side


def cart2pol(epts):
    """ Takes in an Nx2 list of (x,y) points """
    # convert edge points to polar
    x,y = epts[:,0], epts[:,1]
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y,x)
    # combine pairwise the theta and magnitute of points 
    return np.array([t, r]).T


def get_center(mask):
    # calculate moments of binary image
    M = cv2.moments(mask)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cY, cX

# Takes in a grayscale image
def getPieceBitmask(img, showSteps, lt=150, ut=255):
    """ returns a binary mask of the piece"""

    # General purpose kernel used for multiple things

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(5,5))

    # Threshold the image (base 130, 255)
    lower_t = lt
    upper_t = ut
    _, im_th = cv2.threshold(img, lower_t, upper_t, cv2.THRESH_BINARY_INV)
    im_th = cv2.GaussianBlur(im_th, ksize=(15, 15), sigmaX=25)
    im_th = cv2.dilate(im_th, kernel, iterations=2)

    if im_th[0,0] == ut:
        return np.array([]),  False

    # Detect edges within the image
    lower_e = 10
    upper_e = 65
    edges = cv2.Canny(img, upper_e, lower_e)

    # Combine the thresholded image and the edges together
    im_th += edges

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    cv2.dilate(im_floodfill_inv, kernel, iterations=2)

    # Combine the two images to get the foreground.
    imout = im_th | im_floodfill_inv

    # smooth out the image using erosion and dilation
    for i in range(2):
        imout = cv2.erode(imout, kernel, iterations=2 + i)
        imout = cv2.dilate(imout, kernel, iterations=3 + i)

    # convert to greyscale and set
    gray_image = imout.copy()
    gray_image[gray_image < 255] = 0

    # Needed to get rid of extraneous noise in image background
    num_components, grid, stats, centroids = cv2.connectedComponentsWithStats(gray_image)
    max_area_comp = stats[:, -1].argsort()[-2]
    grid[grid != max_area_comp] = 0
    grid[grid == max_area_comp] = 255
    gray_image = grid.astype('uint8')

    if showSteps:
        plt.title("Threshold of ({}, {})".format(lower_t, upper_t))
        plt.imshow(im_th)
        plt.show()

    if showSteps:
        plt.title("Edges of ({}, {})".format(lower_t, upper_t))
        plt.imshow(edges)
        plt.show()

    return gray_image, True


def get_sides(img, img_mask, div_th=20, showSteps=False):
    edges = cv2.morphologyEx(img_mask, cv2.MORPH_GRADIENT, np.ones((7,7))).astype(float)
    edges /= 255.0

    cy, cx = get_center(img_mask)
    center = np.array([cx, cy])

    # [ur, br, ul, bl]
    corners = get_corners(img_mask)

    corners_pol = cart2pol(corners - center)
    corners_pol = np.sort(corners_pol[:, 0])
    ul_corn_t = corners_pol[0]
    ur_corn_t = corners_pol[1]
    br_corn_t = corners_pol[2]
    bl_corn_t = corners_pol[3]

    edges_y, edges_x = np.where(edges > 0)
    edges_coords = np.array([edges_x, edges_y]).T
    edges_theta = cart2pol(edges_coords - center)[:, 0]

    top_edges = edges_coords[(edges_theta <= ur_corn_t) & (edges_theta >= ul_corn_t)]
    bottom_edges = edges_coords[(edges_theta <= bl_corn_t) & (edges_theta >= br_corn_t)]
    right_edges = edges_coords[(edges_theta <= br_corn_t) & (edges_theta >= ur_corn_t)]
    left_edges = edges_coords[(edges_theta <= ul_corn_t) | (edges_theta >= bl_corn_t)]

    puzzle_edges = img.copy()
    puzzle_edges[edges != 1] = [0, 0, 0]

    top_puzzle = puzzle_edges[top_edges[:, 1], top_edges[:, 0]]
    bottom_puzzle = puzzle_edges[bottom_edges[:, 1], bottom_edges[:, 0]]
    right_puzzle = puzzle_edges[right_edges[:, 1], right_edges[:, 0]]
    left_puzzle = puzzle_edges[left_edges[:, 1], left_edges[:, 0]]

    top_y, bot_y, left_x, right_x = top_edges[:, 1], bottom_edges[:, 1], left_edges[:, 0], right_edges[:, 0]
    avg_top_y, std_top_y, max_top_y, min_top_y = np.median(top_y), np.std(top_y), np.max(top_y), np.min(top_y)
    avg_bot_y, std_bot_y, max_bot_y, min_bot_y = np.median(bot_y), np.std(bot_y), np.max(bot_y), np.min(bot_y)
    avg_left_x, std_left_x, max_left_x, min_left_x = np.median(left_x), np.std(left_x), np.max(left_x), np.min(left_x)
    avg_right_x, std_right_x, max_right_x, min_right_x = np.median(right_x), np.std(right_x), np.max(right_x), np.min(right_x)

    top_shape = 0
    bottom_shape = 0
    right_shape = 0
    left_shape = 0
    if std_top_y > div_th:
        if min_top_y < avg_top_y - std_top_y:
            top_shape = 1
        elif max_top_y > avg_top_y + std_top_y:
            top_shape = -1

    if std_bot_y > div_th:
        if max_bot_y > avg_bot_y + std_bot_y:
            bottom_shape = 1
        elif min_bot_y < avg_bot_y - std_bot_y:
            bottom_shape = -1

    if std_left_x > div_th:
        if min_left_x < avg_left_x - std_left_x:
            left_shape = 1
        elif max_left_x > avg_left_x + std_left_x:
            left_shape = -1

    if std_right_x > div_th:
        if max_right_x > avg_right_x + std_right_x:
            right_shape = 1
        elif min_right_x < avg_right_x - std_right_x:
            right_shape = -1

    edges[top_edges[:, 1], top_edges[:, 0]] = 0.3
    edges[bottom_edges[:, 1], bottom_edges[:, 0]] = 0.5
    edges[right_edges[:, 1], right_edges[:, 0]] = 0.7
    edges[left_edges[:, 1], left_edges[:, 0]] = 0.9

    if showSteps:
        plt.imshow(edges, cmap='jet')
        plt.scatter(corners[0, 0], corners[0, 1], color='b')
        plt.scatter(corners[1, 0], corners[1, 1], color='g')
        plt.scatter(corners[2, 0], corners[2, 1], color='y')
        plt.scatter(corners[3, 0], corners[3, 1], color='r')
        plt.scatter(center[0], center[1])
        plt.show()

    return {
        "t": {
            "edge": top_puzzle,
            "shape": top_shape
            },
        "b": {
            "edge": bottom_puzzle,
            "shape": bottom_shape
            },
        "r": {
            "edge": right_puzzle,
            "shape": right_shape
            },
        "l": {
            "edge": left_puzzle,
            "shape": left_shape
            }
    }


def get_corners(img, showSteps=False):

    cy, cx = get_center(img)

    corners = cv2.cornerHarris(img, 10, 3, 0.04)
    max = corner_peaks(corners, min_distance=70, threshold_abs=0.004)

    ul_cand = max[(max[:, 1] < cx) & (max[:, 0] < cy)]
    bl_cand = max[(max[:, 1] < cx) & (max[:, 0] > cy)]
    ur_cand = max[(max[:, 1] > cx) & (max[:, 0] < cy)]
    br_cand = max[(max[:, 1] > cx) & (max[:, 0] > cy)]

    def rectangular_score(ur, br, ul, bl):
        angle_err = abs(ul[0] - ur[0]) + abs(ur[1] - br[1]) + abs(br[0] - bl[0]) + abs(bl[1] - ul[1])
        perim = abs(ul[1] - ur[1]) + abs(ur[0] - br[0]) + abs(br[1] - bl[1]) + abs(bl[0] - ul[0])
        return perim - angle_err

    if showSteps:
        plt.imshow(img)
        plt.scatter(ul_cand[:,1], ul_cand[:, 0])
        plt.scatter(br_cand[:,1], br_cand[:, 0], color='r')
        plt.scatter(ur_cand[:,1], ur_cand[:, 0], color='g')
        plt.scatter(bl_cand[:,1], bl_cand[:, 0], color='b')
        plt.show()

    max_rec = float('inf') * -1
    for ur in ur_cand:
        for br in br_cand:
            for ul in ul_cand:
                for bl in bl_cand:
                    score = rectangular_score(ur, br, ul, bl)
                    if score > max_rec:
                        max_coords = np.array([ur, br, ul, bl])
                        max_rec = score
    if showSteps:
        plt.imshow(img)
        plt.scatter(max_coords[:, 1], max_coords[:, 0])
        plt.show()
    return max_coords[:, [1, 0]]


def load_puzzle():
    pieces = []

    for i in range(8):
        for j in range(13):
            print(i,j)
            f = "images/moanaIndividual/{}_{}.jpg".format(i,j)
            im_in = cv2.imread(f)
            grayscale = cv2.cvtColor(im_in, cv2.COLOR_RGB2GRAY)
            bitmask, _ = getPieceBitmask(grayscale, showSteps=False)
            if i == 7:
                bitmask, _ = getPieceBitmask(grayscale, showSteps=False, lt=130)
            piece_info = get_sides(im_in, bitmask)
            sides = [Side(piece_info[s]["edge"], piece_info[s]["shape"]) for s in piece_info]
            pieces.append(piece(im_in, (i * 8) + j, sides))
    return pieces    


if __name__=='__main__':
    load_puzzle()

