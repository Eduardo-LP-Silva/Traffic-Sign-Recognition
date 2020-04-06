import cv2 as cv
import numpy as np
import math
import utils

# Program's entry point
def main():
    img = utils.readImage()
    img = smooth(img)
    
    img_red, img_blue = img, img

    processImage(img_red, 'Red')
    processImage(img_blue, 'Blue')

    utils.showImage(img, 'Final Classification')

# Find and classify the red or blue signs in an image 
def processImage(img, color):
    img_gray = segment(img, color)
    img_gray = smooth(img_gray) # Post-segmentation smoothing
    img_binary = threshold(img_gray)

    img_binary, max_radius = findContours(img, img_binary, color)
    img_without_objects = cv.bitwise_and(img_binary, img_gray)
    findCircles(img, img_without_objects, max_radius, color)

# Noise smoothing
def smooth(img):
    # Last two values may be adjusted if needed
    return cv.bilateralFilter(img, 5, 75, 75)

# Segments an image by a given color (red or blue)
def segment(img, color):
    # Red Segmentation (HSV Ranges -> (0-179, 0-255, 0-255))
    red_ranges = [(0, 70, 70), (4, 255, 255), (170, 70, 70), (180, 255, 255)]
    # Blue Segmentation
    blue_ranges = [(100, 140, 93), (120, 255, 255)]

    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    if color == 'Red':
        r_mask_1 = cv.inRange(img_hsv, red_ranges[0], red_ranges[1])
        r_mask_2 = cv.inRange(img_hsv, red_ranges[2], red_ranges[3])
        r_mask = r_mask_1 + r_mask_2
        segmented = cv.bitwise_and(img, img, mask=r_mask)
    elif color == 'Blue':
        b_mask = cv.inRange(img_hsv, blue_ranges[0], blue_ranges[1])
        segmented = cv.bitwise_and(img, img, mask=b_mask)

    segmented_gray = cv.cvtColor(segmented, cv.COLOR_BGR2GRAY)

    return segmented_gray

# Converts a grey, segmented image to binary form
def threshold(img_gray):
    _, threshed = cv.threshold(img_gray, 1, 255, cv.THRESH_BINARY)

    return threshed

# Finds contours in a binary image
def findContours(img, img_binary, color):
    #utils.showImage(img_binary)
    areas = []
    mask = img_binary.copy()
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        cnt = contours[i]
        cnt_len = cv.arcLength(cnt, True)

        #Max contour length may be adjusted if needed 
        if(cnt_len <= 70):
            cv.drawContours(mask, [cnt], -1, 0, -1)
            continue

        approx = cv.approxPolyDP(cnt, 0.03 * cv.arcLength(cnt, True), True)

        # Maybe change area so it's a value based on overall image size / area
        # or not cv.isContourConvex(approx) # Add to eliminate further false positives

        height, width = img.shape[:2]
        img_area = width * height
        ratio = int(cv.contourArea(approx)/float(img_area) * 100000.0)
        if ratio <= 65:
            cv.drawContours(mask, [cnt], -1, 0, -1)
            continue

        # if cv.contourArea(approx) < 800:
        #     cv.drawContours(mask, [cnt], -1, 0, -1)
        #     continue

        if(len(approx) <= 4):
            cv.drawContours(mask, [cnt], -1, 0, -1)
        else:
            areas.append(cv.contourArea(cnt))
            continue

        classifyContours(img, approx, color)

    # utils.showImage()
    if(len(areas) != 0):
        max_radius = int(math.sqrt(max(areas) / math.pi))
    else:
        max_radius = 0
    
    return cv.bitwise_and(img_binary, mask), max_radius

# Classifies contours in either triangles or squares/rectangles and displays them over the original image
def classifyContours(img, approx, color):
    font = cv.FONT_HERSHEY_COMPLEX
    approx_ravel = approx.ravel()
    x = approx_ravel[0]
    y = approx_ravel[1]

    side_no = len(approx)
        
    if(side_no <= 4):
        shape = ''

        # TODO: Add more restrictions
        if(side_no == 3):
            side1 = utils.calcDistance(approx_ravel[0], approx_ravel[1], approx_ravel[2], approx_ravel[3])
            side2 = utils.calcDistance(approx_ravel[0], approx_ravel[1], approx_ravel[4], approx_ravel[5])
            side3 = utils.calcDistance(approx_ravel[2], approx_ravel[3], approx_ravel[4], approx_ravel[5])

            diff1 = int(abs(side1 - side2))
            diff2 = int(abs(side1 - side3))
            diff3 = int(abs(side2 - side3))

            if(diff1 > 30 or diff2 > 30 or diff3 > 30):
                return
            
            shape = ' Triangle'
        elif(side_no == 4):
            # print(approx_ravel)
            contour_only_img = np.zeros((len(img), len(img[0])), np.uint8)
            angles = utils.calcCornerAngles(cv.drawContours(contour_only_img, [approx], -1, (255), 1), approx_ravel)

            for i in range(len(angles)):
                if(abs(angles[i] - 90) > 10):
                    return

            shape = ' Rectangle'
        else:
            return

        img = cv.drawContours(img, [approx], -1, (0, 255, 255), 3)
        cv.putText(img, color + shape, (x, y), font, 1, (0, 255, 255), thickness=2)

# Finds circles in a grey image, displaying them over the original one
def findCircles(img, img_binary, max_radius, color):
    #utils.showImage(img_binary)
    font = cv.FONT_HERSHEY_COMPLEX
    rows = img_binary.shape[0]

    if(max_radius == 0):
        return

    tolerance = 5

    # Previous fixed values (for reference): minDist = 70 (?), maxRadius = 50
    # param1 might need to be image specific, evaluate results with fixed 300
    # param1=300, param2=16
    circles = cv.HoughCircles(img_binary, cv.HOUGH_GRADIENT, 1, max_radius * 2 - tolerance, param1=100, param2=16, minRadius=14, maxRadius=max_radius + tolerance)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
      
            cv.circle(img, center, radius, (0, 255, 255), 2)
            cv.putText(img, color + ' Circle', (i[0], i[1] - radius), font, 1, (0, 255, 255), thickness=2)

main()