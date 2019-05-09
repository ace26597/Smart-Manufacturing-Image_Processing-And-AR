# import the necessary packages
import imutils
import numpy as np
import cv2

# load the image, convert it to grayscale, and blur it
image = cv2.imread("images/1.jpg")
image = imutils.resize(image, width=400)
grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find edges in the image using canny edge detection method
# Calculate lower threshold and upper threshold using sigma = 0.33
sigma = 0.33
v = np.median(grayScale)
low = int(max(0, (1.0 - sigma) * v))
high = int(min(255, (1.0 + sigma) * v))

lower_range = np.array([0,100,100], dtype=np.uint8)
upper_range = np.array([10, 255, 255], dtype=np.uint8)

lower_red = np.array([170,100,100])
upper_red = np.array([180,255,255])

edged = cv2.Canny(grayScale, low, high)

# After finding edges we have to find contours
# Contour is a curve of points with no gaps in the curve
# It will help us to find location of shapes

# cv2.RETR_EXTERNAL is passed to find the outermost contours (because we want to outline the shapes)
# cv2.CHAIN_APPROX_SIMPLE is removing redundant points along a line
(_, cnts, _) = cv2.findContours(edged,
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


def detectShape(cnt):
    shape = 'unknown'
    # calculate perimeter using
    peri = cv2.arcLength(c, True)
    # apply contour approximation and store the result in vertices
    vertices = cv2.approxPolyDP(c, 0.04 * peri, True)

    # If the shape it triangle, it will have 3 vertices
    if len(vertices) == 3:
        shape = 'triangle'

    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(vertices) == 4:
        # using the boundingRect method calculate the width and height
        # of enclosing rectange and then calculte aspect ratio

        x, y, width, height = cv2.boundingRect(vertices)
        aspectRatio = float(width) / height

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            shape = "square"
        else:
            shape = "rectangle"

    # if the shape is a pentagon, it will have 5 vertices
    elif len(vertices) == 5:
        shape = "pentagon"

    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"

    # return the name of the shape
    return shape


# Now we will loop over every contour
# call detectShape() for it and
# write the name of shape in the center of image

# loop over the contours
for c in cnts:
    # compute the moment of contour
    M = cv2.moments(c)
    # From moment we can calculte area, centroid etc
    # The center or centroid can be calculated as follows
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    area = cv2.contourArea(c)
    print(area)
    perimeter = cv2.arcLength(c, True)
    print(perimeter)
    # call detectShape for contour c
    shape = detectShape(c)
    # Outline the contours
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(c)  # offsets - with this you get 'mask'
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('cutted contour', image[y:y + h, x:x + w])
    crop_img = image[y:y + h, x:x + w]
    # show the output image

    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    mask0 = cv2.inRange(hsv, lower_range, upper_range)
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    size = crop_img.size / 3
    mask = mask0 + mask1
    print(mask)
    blurred = cv2.GaussianBlur(mask, (11, 11), 0)
    PixelsInRange = cv2.countNonZero(mask)
    print("PixelsInRange", PixelsInRange)
    frac_red = np.divide(float(PixelsInRange), int(size))
    print(frac_red)
    percent_red = np.multiply((float(frac_red)), 100)
    print('Level : ' + str(percent_red) + '%')


    cv2.imshow('mask',mask)

cv2.waitKey(0)
