import cv2 as cv
import numpy as np
import utils

font = cv.FONT_HERSHEY_COMPLEX

# TODO Add option to read image from camera
'''
filename = input('Filename: ')
img = cv.imread(filename)
'''
img = cv.imread('./examples/red_triangle.jpg', cv.IMREAD_COLOR)
#Image Smoothing
img = cv.bilateralFilter(img, 5, 75, 75)
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#Segmentation
#Use simple color segmentation (see Nemo example)
#Red Segmentation (HSV Ranges -> (0-179, 0-255, 0-255))
red_ranges = [(0, 70, 70), (4, 255, 255), (170, 70, 70), (180, 255, 255)]
mask_1 = cv.inRange(img_hsv, red_ranges[0], red_ranges[1])
mask_2 = cv.inRange(img_hsv, red_ranges[2], red_ranges[3])
mask = mask_1 + mask_2
segmented = cv.bitwise_and(img, img, mask=mask)
segmented_gray = cv.cvtColor(segmented, cv.COLOR_BGR2GRAY)
#CHECK Adjust last 2 params
_, threshed = cv.threshold(segmented_gray, 1, 255, cv.THRESH_BINARY)
#CHECK Second argument maybe cv.RETR_CCOMP https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
contours, hierarchy = cv.findContours(threshed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    cnt = contours[i]
    cnt_len = cv.arcLength(cnt, True)

    #CHECK Adjust value
    if(cnt_len <= 70 or hierarchy[0][i][3] != -1):
        continue

    approx = cv.approxPolyDP(cnt, 0.03 * cv.arcLength(cnt, True), True)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    
    if(len(approx) == 3):
        img = cv.drawContours(img, [approx], -1, (255, 0, 0), 3)
        cv.putText(img, 'Triangle', (x, y), font, 1, (0, 0, 255), thickness=2)

utils.showImage(img, 'smt')


#Apply meanshift to relevant areas(?)
#Detect shape in found areas (see video)
'''
window = (300, 200, 100, 50) #(x, y, w, h), (x, y) -> coords of top left corner, (w, h) -> width and height
roi = img
edges = cv.Canny(img,200,255)
cv.imshow('edges', edges)
cv.waitKey(0)
cv.destroyAllWindows()
'''

# Edge detection
# Shape detection -> squares/triangles/circles
# Color detection -> blue/red

# -> Sign detection