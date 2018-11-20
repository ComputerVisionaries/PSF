import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imread, imshow, imsave, electrocardiogram
from  scipy.ndimage.filters import maximum_filter
from scipy.signal import find_peaks


from puzzle.physicalPiece import PhysicalPiece as piece

# accessor functions for sorting
getR = lambda pt : pt[0]
getT = lambda pt : pt[1]

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
    
    return np.array(zip(x,y))

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

        if fourCorners and verticalTest and horizontalTest:
            tcorn = test.copy()

    if tcorn.size == 0:
        return tcorn, False
    
    return tcorn, True

def getSides(bitmask, corners, showSteps):

    # General purpose kernel used for multiple things
    kernel = np.ones((3,3),np.uint8)

    center = getCenter(bitmask)
    edges = cv2.Canny(bitmask,75,12)

    if showSteps:   
        plt.imshow(bitmask)
        plt.scatter(corners[:,1], corners[:,0])
        plt.show()
    
    pet = cart2pol(np.argwhere(edges==255) - center)
    pcorn = cart2pol(corners - center)

    pet = list(pet)
    pet.sort(key=getR)
    pet = np.array(pet)
    
    pcorn = list(pcorn)
    pcorn.sort(key=getR)
    pcorn = np.array(pcorn)

    lr = (pcorn[0][0], pcorn[1][0])
    br = (pcorn[1][0], pcorn[2][0])
    rr = (pcorn[2][0], pcorn[3][0])
    tr = (pcorn[3][0], pcorn[0][0])

    # get the normal edges found
    er = []
    for ran in [lr, br, rr]:
        ranges = pet[(pet[:,0] > ran[0]) & (pet[:,0] < ran[1])]

        tmp = center + pol2cart(ranges)
        er.append(tmp)

    # get the edge found on the boundary between -2pi and 0
    ranges = pet[(pet[:,0] > tr[0]) | (pet[:,0] < tr[1])]
    tmp = center + pol2cart(ranges)
    er.append(np.round(tmp).astype(int))

    er[0][:,1] = (er[0][:,1] * 0.9) + (np.array([center[0]]) * 0.1)
    er[1][:,0] = (er[1][:,0] * 0.9) + (np.array([center[1]]) * 0.1)
    er[2][:,1] = (er[2][:,1] * 0.9) + (np.array([center[0]]) * 0.1)
    er[3][:,0] = (er[3][:,0] * 0.9) + (np.array([center[1]]) * 0.1)


    er[0] = np.round(er[0]).astype(int)
    er[1] = np.round(er[1]).astype(int)
    er[2] = np.round(er[2]).astype(int)
    er[3] = np.round(er[3]).astype(int)
    

    if showSteps:
        for r in er:
            plt.imshow(bitmask)
            plt.scatter(r[:,1], r[:,0])
            plt.show()

    return er, center

def classifySides(sides, center):

    cls = []

    tval = 50

    # classify left
    l = sides[0]   
    lt = l[np.argmin(l[:,0])]
    lb = l[np.argmax(l[:,0])]

    lca = (lt[1] + lb[1]) / 2

    la = sum(l[:,1]) / len(l[:,1])

    print("left")
    print(lca, la)
    if la - lca > tval:
        cls.append(-1)
    elif lca - la > tval:
        cls.append(1)
    else:
        cls.append(0)
        
    # classify bottom
    b = sides[1]
    bl = b[np.argmin(b[:,1])]
    br = b[np.argmin(b[:,1])]
    
    bca = (bl[0] + br[0]) / 2
 
    ba = sum(b[:,0]) / len(b[:,0])

    print("bottom")
    print(bca, ba)
    if bca - ba > tval:
        cls.append(-1)
    elif ba - bca > tval:
        cls.append(1)
    else:
        cls.append(0)

    # classify right
    r = sides[2]
    rt = r[np.argmin(r[:,0])]
    rb = r[np.argmin(r[:,0])]

    rca = (rt[1] + rb[1]) / 2
    
    ra = sum(r[:,1])  / len(r[:,1])

    print("right")
    print(rca, ra)
    if rca - ra > tval:
        cls.append(-1)
    elif ra - rca > tval:
        cls.append(1)
    else:
        cls.append(0)

    # classify top
    t = sides[1]
    tl = t[np.argmin(t[:,1])]
    tr = t[np.argmin(t[:,1])]
    
    tca = (tl[0] + tr[0]) / 2
    
    ta = sum(t[:,0]) / len(t[:,0])

    print("top")
    print(tca, ta)
    if ta - tca > tval:
        cls.append(-1)
    elif tca - ta > tval:
        cls.append(1)
    else:
        cls.append(0) 

    print(cls)

    
    


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
            corners, cornerSuccess = getCorners(bitmask, ival, 1, showSteps)
            #print("Testing tval: {} output {}".format(ival, success))
            ival *= 1.5

            if ival > 100:
                break

        # Loop to find a corner assignment by varying the max threshold value in a different way
        bfactor = .9
        while not cornerSuccess and bfactor >= .5:
            corners, cornerSuccess = getCorners(bitmask, ival, bfactor, showSteps)
            bfactor -= .05

        # failed to find a corner assignment so lower bitmask threshold and try again
        if not cornerSuccess:
            continue

        # show the successful corner assignment
        if showSteps:
            plt.imshow(img)
            plt.title("Points on piece")
            plt.scatter(corners[:,1], corners[:,0], c='#FF0000')
            plt.show()

        break

    if lower < 100:
        return False

    sidePts, center = getSides(bitmask, corners, showSteps)
    #sideType = classifySides(sidePts, center)

    # get pixel values of side points

    
    evals = {}
    evals["l"] = [img[pt[0],pt[1]] for pt in sidePts[0]]
    evals["b"] = [img[pt[0],pt[1]] for pt in sidePts[1]]
    evals["r"] = [img[pt[0],pt[1]] for pt in sidePts[2]]
    evals["t"] = [img[pt[0],pt[1]] for pt in sidePts[3]]

    return evals


def generate_edges():
    pieces = []

    for i in range(8):
        for j in range(13):
            print(i,j)
            f = "images/moanaIndividual/{}_{}.jpg".format(i,j)
            im_in = cv2.imread(f)
            success = extractPiece(im_in, False)

            if success == False:
                raise Exception("Failed to extract the piece")

            else:
                #pieces.append(piece(im_in, (i * 8) + j, edges))
                np.save("edges/{}_{}.npy".format(i,j), success)

def load_puzzle():
    pieces = []

    for i in range(8):
        for j in range(13):
            print(i,j)
            f = "images/moanaIndividual/{}_{}.jpg".format(i,j)
            im_in = cv2.imread(f)
            edges = np.load('edges/{}_{}.npy'.format(i,j)).item()
            
            pieces.append(piece(im_in, (i * 8) + j, edges))

    


