import cv2 as cv

def readImage():
    # TODO Add option to read image from camera
    #filename = input('Filename: ')
    img = cv.imread('./examples/red_circle.jpg', cv.IMREAD_COLOR) #TODO Replace with var filename
    
    return img

def showImage(image, windowName='OpenCV'):
    cv.imshow(windowName, image)
    cv.waitKey(0)
    cv.destroyAllWindows()