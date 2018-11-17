import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imread, imshow, imsave, electrocardiogram
from  scipy.ndimage.filters import maximum_filter
from scipy.signal import find_peaks

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def getCenter(mask):
    # calculate moments of binary image
    M = cv2.moments(mask)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cY, cX

def getOutline(img, showSteps=False):

    kernel = np.ones((3,3),np.uint8)

    # Threshold the image
    lower_t = 155
    upper_t = 255
    _, im_th = cv2.threshold(img, lower_t, upper_t, cv2.THRESH_BINARY_INV);

    if showSteps:
        plt.title("Threshold of ({}, {})".format(lower_t, upper_t))
        plt.imshow(im_th)
        plt.show()


    # Detect edges within the image
    lower_e = 12
    upper_e = 75
    edges = cv2.Canny(img, upper_e, lower_e)

    if showSteps:
        plt.title("Edges of ({}, {})".format(lower_t, upper_t))
        plt.imshow(edges)
        plt.show()
    
    # Combine the thresholded image and the edges together
    for i in range(3):
        im_th[:,:,i] += edges


    # Copy the thresholded image.
    im_floodfill = im_th.copy()
 
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
     
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    cv2.dilate(im_floodfill_inv, kernel, iterations=2)
     
    # Combine the two images to get the foreground.
    imout = im_th | im_floodfill_inv

    # smooth out the image using erosion and dialation
    for i in range(2):
        imout = cv2.erode(imout, kernel, iterations = 3 + i)
        imout = cv2.dilate(imout, kernel, iterations = 2 + i)

    # convert to greyscale and set
    gray_image = cv2.cvtColor(imout, cv2.COLOR_BGR2GRAY)
    gray_image[gray_image < 255] = 0

    #imshow(gray_image)
    if showSteps:
        plt.title("Piece Bitmask")
        plt.imshow(gray_image)
        plt.show()


    center = getCenter(gray_image)

    edges = cv2.Canny(gray_image,75,12)

    if showSteps:
        plt.imshow(edges)
        plt.scatter(center[1], center[0])
        plt.title("Center of Piece found")
        plt.show()

    # center the edge points
    epts = np.nonzero(edges)
    epts = np.array(zip(epts[0],epts[1]))
    epts = epts - center

    # convert edge points to polar
    x,y = epts[:,0], epts[:,1]
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y,x)

    # combine pairwise the theta and magnitute of points 
    epts = zip(t,r)

    # accessor functions for sorting
    getR = lambda pt : pt[0]
    getT = lambda pt : pt[1]

    # sort edge points by angle
    epts.sort(key=getR)

    # I convert from python list to numpy array often
    # - Python list sort method does what I want
    # - Numpy is used to access and do stuff with 
    epts = np.array(epts) 

    if showSteps:
        plt.scatter(epts[:,0], epts[:,1])
        plt.title("Polar Edge points")
        plt.show()  

    # I take the average distance to the center
    # This is used as a threshold to find the corner points of the image
    average = sum(epts[:,1]) / epts.shape[0]

    # Get all points above average
    outer = epts[epts[:,1] > average]
    

    # Sort theshholded edge points by magnitute
    tse = list(outer.copy())
    tse.sort(key=getT)
    tse = np.array(tse)

    # Get the 100 smallest magnitute edge points
    # These are used to gind the groups of interest to try and find the corners
    # bpoints are sorted by radius to work around in a circle
    bpoints = tse[0:100]
    bpoints = list(bpoints)
    bpoints.sort(key=getR)

    # TODO fix the bellow code to go through and find the corners better
    cpatches = [bpoints[0]]
    ranges = []
    cpoint = bpoints[0]
    npoint = bpoints[1]
    master = 0
    rctr = cpoint[0]

    for pt in bpoints[1:]:
        ctr = sum((tse[:,0] < npoint[0]) & (tse[:,0] > cpoint[0]))
        if (ctr - master) > 5:
           cpatches.append(pt)
        rjump = abs(rctr- npoint[0])
        if (rjump > .3):
            cpatches.append(pt)
        #print(rjump)
        master = ctr
        rctr = npoint[0]
        npoint = pt

    cpatches = np.array(cpatches)

    if showSteps:
        plt.title("Points of Interest")
        plt.scatter(outer[:,0], outer[:,1])
        plt.scatter(cpatches[:,0], cpatches[:,1], c='#FF0000')
        plt.show()

    # TODO 
    """
    1. Select points for ranges of interest (basically the front and the back of t the points above a threshold for each corner"
    2. Get the max of the points within the range of interest
    3. Determine if the range of interest signifies a corner, a piece head, or a flat side (potential but not likely)
    4. Sample points from the edge of the piece to act as the side feature
    """  



if __name__ == "__main__":
    im_in = cv2.imread("images/moanaIndividual/3_8.jpg")
    #im_in = cv2.imread("images/moanaIndividual/0_0.jpg")

    getOutline(im_in, True)


