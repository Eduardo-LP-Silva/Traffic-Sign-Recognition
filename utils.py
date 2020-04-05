import cv2 as cv
import math

# Reads an image in colored mode
def readImage():
    # TODO Add option to read image from camera
    #filename = input('Filename: ')
    img = cv.imread('./examples/1.jpg', cv.IMREAD_COLOR) #TODO Replace with var filename
    
    return img

# Shows an image and waits for user input before destroying the respective window
def showImage(image, windowName='OpenCV'):
    cv.imshow(windowName, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def calcDistance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

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