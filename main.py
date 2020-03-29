import cv2 as cv
import numpy as np
import utils

def main():
    img = utils.readImage()
    img = smooth(img)

    img_gray, r_mask = segment(img)
    masks = [r_mask] #TODO Add blue mask
    img_binary = threshold(img_gray)

    displayContours(img, img_binary, masks)

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

def displayContours(img, img_binary, masks):
    font = cv.FONT_HERSHEY_COMPLEX
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

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
            color = determineColor(masks, y, x)
            cv.putText(img, color + ' Triangle', (x, y), font, 1, (0, 0, 255), thickness=2)

def determineColor(masks, y, x):
    if(masks[0][y][x] == 255):
        return 'Red'

main()