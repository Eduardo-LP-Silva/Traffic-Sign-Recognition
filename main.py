import cv2 as cv
import numpy as np

# TO DO: option to take the picture
# read from file
filename = input('Filename: ')
img = cv.imread(filename)

# img = cv.imread('./examples/blue-rectangles/1.jpg') # retorna Mat

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

# applying Canny edge detector
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_edges_enhanced = cv.Canny(img, 150, 200)

cv.imshow('Canny edge detector', img_edges_enhanced)
cv.waitKey(0)
cv.destroyAllWindows()

# detecting contours
# contours, hierarchy = cv.findContours(img_edges_enhanced, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv.findContours(img_edges_enhanced, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

img_contours = cv.drawContours(img, contours, -1, (0,255,0), 1)
cv.imshow('Image contours', img_contours)
cv.waitKey(0)
cv.destroyAllWindows()