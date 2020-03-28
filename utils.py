import cv2 as cv

def showImage(image, windowName):
    cv.imshow(windowName, image)
    cv.waitKey(0)
    cv.destroyAllWindows()