import cv2 as cv
import numpy as np
import utils

def main():
    img = utils.readImage()
    img = smooth(img)

    img_gray, r_mask = segment(img)
    masks = [r_mask] #TODO Add blue mask
    img_gray = smooth(img_gray)
    img_binary = threshold(img_gray)

    findCircles(img, img_gray, img_binary)
    #findContours(img, img_binary, masks)

    #utils.showImage(img_gray)
    utils.showImage(img, 'Final Classification')

def smooth(img):
    return cv.bilateralFilter(img, 5, 75, 75)

def segment(img):
    #Red Segmentation (HSV Ranges -> (0-179, 0-255, 0-255))
    red_ranges = [(0, 70, 70), (4, 255, 255), (170, 70, 70), (180, 255, 255)]

    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    r_mask_1 = cv.inRange(img_hsv, red_ranges[0], red_ranges[1])
    r_mask_2 = cv.inRange(img_hsv, red_ranges[2], red_ranges[3])
    r_mask = r_mask_1 + r_mask_2
    segmented = cv.bitwise_and(img, img, mask=r_mask)
    segmented_gray = cv.cvtColor(segmented, cv.COLOR_BGR2GRAY)

    return segmented_gray, r_mask

def threshold(img_gray):
    #CHECK Adjust last 2 params
    _, threshed = cv.threshold(img_gray, 1, 255, cv.THRESH_BINARY)

    return threshed

def findContours(img, img_binary, masks):
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        cnt = contours[i]
        cnt_len = cv.arcLength(cnt, True)

        #CHECK Adjust value
        if(cnt_len <= 70 or hierarchy[0][i][3] != -1):
            continue

        approx = cv.approxPolyDP(cnt, 0.03 * cv.arcLength(cnt, True), True)
        classifyContours(img, approx, masks)

def findCircles(img, img_gray, img_binary):
    font = cv.FONT_HERSHEY_COMPLEX
    rows = img_gray.shape[0]

    max = 0

    for i in range(len(img_binary)):
        line_max = 0
        line_aux = 0

        for j in range(len(img_binary[i])):
            if img_binary[i][j] != 0:
                line_aux += 1
            else:
                if line_max < line_aux:
                    line_max = line_aux
                line_aux = 0

        if max < line_max:
            max = line_max
    
    #Fixed values: minDist = 70 (?), maxRadius = 50
    circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, 1, max, param1=300, param2=14, minRadius=14, maxRadius=max)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]

            cv.circle(img, center, radius, (0, 255, 255), 2)
            #cv.putText(img, color + ' Circle', center, font, 1, (0, 0, 255), thickness=2)

def classifyContours(img, approx, masks):
    font = cv.FONT_HERSHEY_COMPLEX

    x = approx.ravel()[0]
    y = approx.ravel()[1]

    side_no = len(approx)
        
    if(side_no <= 4):
        img = cv.drawContours(img, [approx], -1, (0, 255, 255), 3)
        color = findColor(masks, y, x)
        shape = ''

        if(side_no == 3):
            shape = ' Triangle'
        elif(side_no == 4):
            shape = ' Rectangle'

        cv.putText(img, color + shape, (x, y), font, 1, (0, 0, 255), thickness=2)

def findColor(masks, y, x):
    if(masks[0][y][x] == 255):
        return 'Red'

main()