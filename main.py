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
    
    return shape


# TO DO: option to take the picture
# read from keyboard the name of the image
# filename = input('Filename: ')
# img = cv.imread(filename)

img = cv.imread('./examples/blue-rectangles/1.jpg') # retorna Mat
display_image('Original image', img)

###### COLOR #######
# Maybe change to HSV?

# shades of blue (light and dark)
light_blue = (255, 150, 60)
dark_blue = (100, 0, 0)

# all values that are not in the above range will be black
mask = cv.inRange(img, dark_blue, light_blue)
result = cv.bitwise_and(img, img, mask=mask)

# display_image('Thresholding blue', result)

###### SHAPE #######

# applying Canny edge detector
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray_img, (5, 5), 0)
# display_image('Blurred', blurred)
img_edges_enhanced = cv.Canny(blurred, 100, 200)

# show the result of applying Canny
# display_image('Canny edge detector', img_edges_enhanced)

# detecting contours
# contours, hierarchy = cv.findContours(img_edges_enhanced, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours, _ = cv.findContours(img_edges_enhanced, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)    

# show the result contours (60)
# img_contours = cv.drawContours(img, contours, -1, (0,255,0), 1)

# show the image contours
# display_image('Image contours (in green)', img_contours)

# loop over the contours
for c in contours:
    # compute the center of the contour, then detect the name of the shape using only the contour
    M = cv.moments(c)
    # cX = int((M["m10"] / M["m00"]) * ratio)
    # cY = int((M["m01"] / M["m00"]) * ratio)
    if M['m00'] == 0:
        continue

    shape = detect_shape(c)
    print(shape)
    if shape != 'rectangle' and shape != 'square':
        continue

    cv.drawContours(img, [c], -1, (0, 255, 0), 2)
    cX = c[0,0,0]
    cY = c[0,0,1]
    cv.putText(img, shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # display_image('Shape', img)

display_image('Shapes', img)