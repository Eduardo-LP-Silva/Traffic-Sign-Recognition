import cv2 as cv
import numpy as np

def display_image(window_name, img):
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def detect_shape(contour):
    shape = 'unidentified'
    # parameters: curve, closed
    peri = cv.arcLength(contour, True) # perimeter
    # parameters: curve, epsilon (approximation accuracy), closed
    approx = cv.approxPolyDP(contour, 0.04 * peri, True)
    # if the shape has 4 vertices, it is either a square or a rectangle
    if len(approx) == 4:
        # compute the bounding box of the contour and use it to compute the aspect ratio
        (x, y, w, h) = cv.boundingRect(approx)
        ar = w / float(h)

        # a square will have an aspect ratio that is approximately equal to one, otherwise, the shape is a rectangle
        shape = 'square' if ar >= 0.95 and ar <= 1.05 else 'rectangle'
    
    return shape, approx


# TO DO: option to take the picture
# read from keyboard the name of the image
filename = input('Filename: ')
img = cv.imread(filename)

# img = cv.imread('./examples/blue-rectangles/3.jpg') # retorna Mat
display_image('Original image', img)
blurred = cv.bilateralFilter(img, 5, 75, 75)
# display_image('Blurred', blurred)

###### COLOR #######
# Maybe change to HSV?

# shades of blue (light and dark)
light_blue = (255, 150, 60)
dark_blue = (100, 0, 0)

# all values that are not in the above range will be black
mask = cv.inRange(blurred, dark_blue, light_blue)
result = cv.bitwise_and(blurred, blurred, mask=mask)

# display_image('Thresholding blue', result)

###### SHAPE #######

gray_img = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
_, bynary_img = cv.threshold(gray_img, 1, 255, cv.THRESH_BINARY)

# detecting contours
contours, _ = cv.findContours(bynary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)    

# loop over the contours
for c in contours:
    # compute the center of the contour, then detect the name of the shape using only the contour
    M = cv.moments(c)
    if M['m00'] == 0:
        continue
    
    shape, c = detect_shape(c)
    if shape != 'rectangle' and shape != 'square':
        continue

    if cv.contourArea(c) < 1000:
        continue

    cv.drawContours(img, [c], -1, (0, 255, 0), 2)
    cX = c[0,0,0]
    cY = c[0,0,1]
    cv.putText(img, shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

display_image('Shapes', img)