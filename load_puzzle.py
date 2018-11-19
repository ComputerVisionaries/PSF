import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imread, imshow, imsave, electrocardiogram
from  scipy.ndimage.filters import maximum_filter
from scipy.signal import find_peaks

def cart2pol(epts):
    """ Takes in an Nx2 list of (x,y) points """
    # convert edge points to polar
    x,y = epts[:,0], epts[:,1]
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y,x)

    # combine pairwise the theta and magnitute of points 
    return zip(t,r)

def pol2cart(pts):
    """ Takes in an Nx2 list of (theta,magnitude) points """
    x = np.cos(pts[:,0]) * pts[:,1]
    y = np.sin(pts[:,0]) * pts[:,1]
    
    return zip(x,y)

def getCenter(mask):
    # calculate moments of binary image
    M = cv2.moments(mask)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cY, cX

def getPieceBitmask(img, showSteps, lt=150, ut=240):
    """ returns a binary mask of the piece"""

    # General purpose kernel used for multiple things
    kernel = np.ones((3,3),np.uint8)

    # Threshold the image (base 150, 240)
    lower_t = lt
    upper_t = ut
    _, im_th = cv2.threshold(img, lower_t, upper_t, cv2.THRESH_BINARY_INV);


    if im_th[0,0] == ut:
        return np.array([]),  False

    # Detect edges within the image
    lower_e = 12
    upper_e = 75
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
    gray_image = imout.copy()#cv2.cvtColor(imout, cv2.COLOR_BGR2GRAY)
    gray_image[gray_image < 255] = 0

    tltest = gray_image[0,0] == 255
    trtest = gray_image[0,-1] == 255
    bltest = gray_image[-1,0] == 255    
    brtest = gray_image[-1,-1] == 255

    if tltest or trtest or bltest or brtest:
        return np.array([]),  False

    if showSteps:
        plt.title("Threshold of ({}, {})".format(lower_t, upper_t))
        plt.imshow(im_th)
        plt.show()

    if showSteps:
        plt.title("Edges of ({}, {})".format(lower_t, upper_t))
        plt.imshow(edges)
        plt.show()
    
    #gray_image = cv2.erode(gray_image, kernel, iterations = 6)
    return gray_image, True

def getCorners(img, tval, bfactor=1, showSteps=True):

    # General purpose kernel used for multiple things
    kernel = np.ones((3,3),np.uint8)

    center = getCenter(img)

    edges = cv2.Canny(img,75,12)

    if showSteps:
        plt.imshow(edges)
        plt.scatter(center[1], center[0])
        plt.title("Center of Piece found")
        plt.show()

    # center the edge points
    epts = np.nonzero(edges)
    epts = np.array(zip(epts[0],epts[1]))
    epts = epts - center

    cpts = epts.copy()

    # convert edge points to polar
    epts = cart2pol(epts)

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
        print(tval)
        plt.scatter(epts[:,0], epts[:,1])
        plt.title("Polar Edge points")
        plt.show()  

    
    # I take the average distance to the center
    # This is used as a threshold to find the corner points of the image  
    mx = max(epts[:,1])    
    average = sum(epts[:,1]) / epts.shape[0]

    average = (average * bfactor) + mx/tval

    # Get all points above average
    outer = epts[epts[:,1] > average]

    if showSteps:
        plt.title("All Bpoints")
        plt.scatter(outer[:,0], outer[:,1])
        plt.show()
    
    # Sort theshholded edge points by magnitute
    tse = list(outer.copy())
    tse.sort(key=getT)
    tse = np.array(tse)

    mx = 0
    t = []
    for i in range(len(outer) - 1):
        l = outer[i]
        r = outer[i + 1]

        rd = r[0] - l[0]
        if rd > .1:
            t.append((l,r))

    t = np.array(t)

    if t.size == 0:
        print("Error : no points of interest found in outer")
        return False, []
    
    if showSteps:

        rad = t[:,:,0]
        mag = t[:,:,1]

        plt.title("Found boundaries")
        plt.scatter(outer[:,0], outer[:,1])
        plt.scatter(rad, mag, c='#FF0000')
        plt.show()


    maxs = []

    for i in range(len(t) - 1):
        l = t[i][1]
        r = t[i + 1][0]

        pts = outer[(outer[:,0] > l[0]) & (outer[:,0] < r[0])]
        #if pts.size == 0:
        if pts.size == 0:
            continue
        m = max(pts, key=getT)
        maxs.append(m)

    # get one of the corners
    pts = outer[outer[:,0] < t[0][0][0]]
    if pts.size == 0:
        return [], False
    omx = max(pts, key=getT)
    
    pts = outer[outer[:,0] > t[-1][1][0]]
    if pts.size == 0:
        return [], False
    tmx = max(pts, key=getT)

    maxs.append(omx)

    maxs.append(tmx)
    
    maxs = np.array(maxs)

    if showSteps:
        plt.title("Found maxes")
        plt.scatter(outer[:,0], outer[:,1])
        plt.scatter(maxs[:,0], maxs[:,1], c='#FF0000')
        plt.show()

    # edges
    corners = pol2cart(maxs)
    corners = np.array(corners)


    #if showSteps:
    opts = cpts + center
    ocrn = corners + center

    if showSteps:
        plt.imshow(img)
        plt.title("Points on piece")
        plt.scatter(ocrn[:,1], ocrn[:,0], c='#FF0000')
        plt.show()

    tcorn = np.array([])
    for cand in ocrn:
        test = [cand]
        ftest = []
        for corn in ocrn:
            if not (cand[0] == corn[0] and cand[1] == corn[1]):
                latd = abs(cand[1] - corn[1])
                verd = abs(cand[0] - corn[0])

                if latd < 100 or verd < 100:
                    test.append(corn)
                else:
                    ftest.append(corn)
        
        if len(test) == 3:
            c0 = test[0]
            c1 = test[1]
            c2 = test[2]

            mn = 200
            mc = None

            for corn in ftest:
                t1 = min(abs(c1[0] - corn[0]), abs(c1[1] - corn[1]))
                t2 = min(abs(c2[0] - corn[0]), abs(c2[1] - corn[1]))

                if (t1 < 100 and t2 < 100) and (t1 + t2 < mn):
                    mn = t1 + t2
                    mc = corn

                
            if not mc is None:
                test.append(mc)


        test = np.array(test)

        moreCorners = len(test) > len(tcorn)
        fourCorners = len(test) == 4

        verticalTest = (sum(test[:,0] > center[0]) == 2)
        horizontalTest = (sum(test[:,1] > center[1]) == 2)

        spaceTest = True
        for pt in test:
            if spaceTest:
                #print(pt, (test - pt > 0) & (test - pt < 100))
                spaceTest = not np.any((test - pt > 0) & (test - pt < 100))


        if fourCorners and verticalTest and horizontalTest:
            tcorn = test.copy()

    if tcorn.size == 0:
        return tcorn, False
    return tcorn, True


def extractPiece(img, showSteps=False):

    # Show image if debug on
    if showSteps:
        imshow(img)
    
    # Loop from a threshold value of 200 to 100 to try and find the corners
    lower = 170
    while lower >= 100:

        # Bitmap extraction works better on grayscale image
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        bitmaskSuccess = False
        bitmask = np.array([])
        while not bitmaskSuccess and lower >= 0:
            #print(lower)
            bitmask, bitmaskSuccess = getPieceBitmask(grayImg, showSteps, lower)
            lower -= 2

        # No acceptable bitmask found. Lower threshold and try again
        if not bitmaskSuccess:
            continue

        if showSteps:
            plt.title("Piece Bitmask")
            plt.imshow(bitmask)
            plt.show()

        # Loop to find a corner assignment by varying max threshold value 
        ival = 8
        cornerSuccess = False
        while not cornerSuccess:
            tcorn, cornerSuccess = getCorners(bitmask, ival, 1, showSteps)
            #print("Testing tval: {} output {}".format(ival, success))
            ival *= 1.5

            if ival > 100:
                break

        # Loop to find a corner assignment by varying the max threshold value in a different way
        bfactor = .9
        while not cornerSuccess and bfactor >= .5:
            tcorn, cornerSuccess = getCorners(bitmask, ival, bfactor, showSteps)
            bfactor -= .05

        # failed to find a corner assignment so lower bitmask threshold and try again
        if not cornerSuccess:
            continue

        # show the successful corner assignment
        plt.imshow(img)
        plt.title("Points on piece")
        plt.scatter(tcorn[:,1], tcorn[:,0], c='#FF0000')
        plt.show()

        return True
    if lower < 100:
        return False



if __name__ == "__main__":
    """
    for i in range(8):
        for j in range(12,13):
            print(i,j)
            f = "images/moanaIndividual/{}_{}.jpg".format(i,j)
            im_in = cv2.imread(f)
            success = extractPiece(im_in, False)

            if not success:
                print("error test {} failed".format(i))
    """
    im_in = cv2.imread("images/moanaIndividual/1_12.jpg")
    success = False
    success = extractPiece(im_in, False)
    print(success)


