import cv2 as cv
import numpy as np

# TO DO: option to take the picture
# read from file
#filename = input('Filename: ')
#img = cv.imread(filename)

img = cv.imread('./examples/blue-rectangles/1.jpg') # retorna Mat
# hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

cv.imshow('Traffic signs', img)
cv.waitKey(0)
cv.destroyAllWindows()

# shades of blue (light and dark)
light_blue = (255, 150, 60)
dark_blue = (100, 0, 0)

# all values that are not in the above range will be black
mask = cv.inRange(img, dark_blue, light_blue)
result = cv.bitwise_and(img, img, mask=mask)

# show the result
cv.imshow('Thresholding blue', result)
cv.waitKey(0)
cv.destroyAllWindows()
