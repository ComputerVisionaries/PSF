import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imread, imshow, imsave

im_in = cv2.imread("images/moanaIndividual/5_5.jpg")

# Threshold.
# Set values equal to or above 220 to 0.
# Set values below 220 to 255.

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
 
th, im_th = cv2.threshold(im_in, 150, 255, cv2.THRESH_BINARY_INV);
edges = cv2.Canny(im_in,75,12)
kernel = np.ones((3,3),np.uint8)

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
for i in range(2):
    imout = cv2.erode(imout, kernel, iterations = 3 + i)
    imout = cv2.dilate(imout, kernel, iterations = 2 + i)

gray_image = cv2.cvtColor(imout, cv2.COLOR_BGR2GRAY)
gray_image[gray_image == 226] = 0

# calculate moments of binary image
M = cv2.moments(gray_image)

# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])


center = (cY, cX)

#cv2.circle(gray_image, (cX, cY), 5, (0, 255, 0), -1)
#cv2.putText(gray_image, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#imshow(gray_image)

edges = cv2.Canny(gray_image,75,12)
epts = np.nonzero(edges)
epts = np.array(zip(epts[0],epts[1]))
epts = epts - center

x,y = epts[:,0], epts[:,1]
r = np.sqrt(x**2+y**2)
t = np.arctan2(y,x)

epts = np.array([r,t])

plt.scatter(epts[1], epts[0])
plt.show()



"""
for x,y in zip(epts[0],epts[1]):
    test[x,y] = 255
print("test")
imshow(test)
"""

"""
test = cv2.imread("images/moanaIndividual/5_0.jpg", cv2.IMREAD_GRAYSCALE) 

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 1000000
params.maxArea = 10000000
params.filterByCircularity = False
params.filterByColor = False
params.filterByConvexity = False
params.filterByInertia = False
# blob detection only works with "uint8" images.
params.minThreshold = int(0)
params.maxThreshold = int(255)
params.thresholdStep = 15

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(test)

output = cv2.drawKeypoints(test, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

imshow(output)
"""

"""
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

imshow(unknown)
"""
