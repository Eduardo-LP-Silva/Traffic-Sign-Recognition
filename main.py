import cv2 as cv
import numpy as np
import utils

# Program's entry point
def main():
    img = utils.readImage()
    img = smooth(img)

    processImage(img, 'Red')
    #TODO Call processImage again with color 'Blue'

    utils.showImage(img, 'Final Classification')

# Find and classify the red or blue signs in an image 
def processImage(img, color):
    img_gray, mask = segment(img, color)
    img_gray = smooth(img_gray) #Post-segmentation smoothing
    img_binary = threshold(img_gray)

    findCircles(img, img_gray, img_binary, color)
    findContours(img, img_binary, color)

# Smooths an image
def smooth(img):
    #Last two values may be adjusted if needed
    return cv.bilateralFilter(img, 5, 75, 75)

# Segments an image by a given color (red or blue)
def segment(img, color):
    #Red Segmentation (HSV Ranges -> (0-179, 0-255, 0-255))
    red_ranges = [(0, 70, 70), (4, 255, 255), (170, 70, 70), (180, 255, 255)]
    #TODO Add blue segmentatio ranges

    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    if color == 'Red':
        r_mask_1 = cv.inRange(img_hsv, red_ranges[0], red_ranges[1])
        r_mask_2 = cv.inRange(img_hsv, red_ranges[2], red_ranges[3])
        r_mask = r_mask_1 + r_mask_2
        segmented = cv.bitwise_and(img, img, mask=r_mask)
        segmented_gray = cv.cvtColor(segmented, cv.COLOR_BGR2GRAY)
    #TODO Add Blue condition

    return segmented_gray, r_mask

# Converts a grey, segmented image to binary form
def threshold(img_gray):
    _, threshed = cv.threshold(img_gray, 1, 255, cv.THRESH_BINARY)

    return threshed

# Finds contours in a binary image
def findContours(img, img_binary, color):
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        cnt = contours[i]
        cnt_len = cv.arcLength(cnt, True)

        #Max contour length may be adjusted if needed 
        if(cnt_len <= 70 or hierarchy[0][i][3] != -1):
            continue

        approx = cv.approxPolyDP(cnt, 0.03 * cv.arcLength(cnt, True), True)
        classifyContours(img, approx, color)

# Finds circles in a grey image, displaying them over the original one
def findCircles(img, img_gray, img_binary, color):
    font = cv.FONT_HERSHEY_COMPLEX
    rows = img_gray.shape[0]

    max = utils.getMaxCircleWidth(img_binary)
    
    #Previous fixed values (for reference): minDist = 70 (?), maxRadius = 50
    #param1 might need to be image specific, evaluate results with fixed 300
    circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, 1, max, param1=300, param2=14, minRadius=14, maxRadius=max)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
      
            cv.circle(img, center, radius, (0, 255, 255), 2)
            cv.putText(img, color + ' Circle', (i[0] + radius, i[1] + radius), font, 1, (0, 255, 255), thickness=2)

# Classifies contours in either triangles or squares/rectangles and displays them over the original image
def classifyContours(img, approx, color):
    font = cv.FONT_HERSHEY_COMPLEX

    x = approx.ravel()[0]
    y = approx.ravel()[1]

    side_no = len(approx)
        
    if(side_no <= 4):
        img = cv.drawContours(img, [approx], -1, (0, 255, 255), 3)
        shape = ''

        #TODO Add more restrictions
        #Equilateral sort of
        #60º angles
        if(side_no == 3):
            shape = ' Triangle'
        #90º angle between sides
        #Area minimum threshold
        elif(side_no == 4):
            shape = ' Rectangle'

        cv.putText(img, color + shape, (x, y), font, 1, (0, 255, 255), thickness=2)

main()