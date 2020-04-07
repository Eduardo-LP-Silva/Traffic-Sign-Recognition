import cv2 as cv
import numpy as np
import math

# Reads an image in colored mode
def readImage():
    # TODO Add option to read image from camera
    # filename = input('Filename: ')
    img = cv.imread('./examples/rectangles/1.jpg', cv.IMREAD_COLOR) # TODO Replace with var filename
    return img

def saveImage(img, filename='example.png'):
    cv.imwrite(filename, img)

# Shows an image and waits for user input before destroying the respective window
def showImage(image, windowName='OpenCV'):
    cv.imshow(windowName, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def calcDistance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

def calcCornerAngles(cnt_img, corners):
    angles = []
    d1 = int(calcDistance(corners[0], corners[1], corners[2], corners[3]))
    d2 = int(calcDistance(corners[2], corners[3], corners[4], corners[5]))
    d3 = int(calcDistance(corners[4], corners[5], corners[6], corners[7]))
    d4 = int(calcDistance(corners[6], corners[7], corners[0], corners[1]))

    max_radius = min([d1, d2, d3, d4])

    for k in range(len(corners) - 1):
        if(k % 2 != 0):
            continue

        blank_img = np.zeros((len(cnt_img), len(cnt_img[0])), np.uint8)
        cv.circle(blank_img, (corners[k], corners[k + 1]), max_radius // 2, (255, 255, 255))
        intersect_img = cv.bitwise_and(cnt_img, blank_img)

        intersect_pts = np.where(intersect_img > 1)

        # print(intersect_pts)
        # showImage(cv.circle(cnt_img, (corners[k], corners[k + 1]), max_radius // 2, (255, 255, 255)))
        # print(corners[k + 1])
        
        if(len(intersect_pts[0]) < 2 or len(intersect_pts[1]) < 2):
            angles.append(0)
            continue

        vector1 = (intersect_pts[1][0] - corners[k], intersect_pts[0][0] - corners[k + 1])
        vector2 = (intersect_pts[1][1] - corners[k], intersect_pts[0][1] - corners[k + 1])

        # print(vector1)
        # print(vector2)

        scalar_p = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        norm1 = math.sqrt(math.pow(vector1[0], 2) + math.pow(vector1[1], 2))
        norm2 = math.sqrt(math.pow(vector2[0], 2) + math.pow(vector2[1], 2))

        angle = math.acos(scalar_p / (norm1 * norm2)) * 180 / math.pi
        angles.append(angle)

    return angles


# Returns the maximum number of joined object pixels
def getMaxCircleWidth(img_binary):
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

    return max