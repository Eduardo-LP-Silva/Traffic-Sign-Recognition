import cv2 as cv
import numpy as np

# It's missing the option to take the picture
# Read from file
#filename = input('Filename: ')
#img = cv.imread(filename)
img = cv.imread('./examples/exemplo1.jpg', 0)
cv.imshow('Traffic signs', img)
cv.waitKey(0)
cv.destroyAllWindows()

edges = cv.Canny(img,200,255)
cv.imshow('edges', edges)
cv.waitKey(0)
cv.destroyAllWindows()

# Is noise smoothing necessary?
# Edge detection
# Shape detection -> squares/triangles/circles
# Color detection -> blue/red

# -> Sign detection
