import cv2 as cv
import numpy as np

image = cv.imread("00000_00000_00000.png")
image = cv.resize(image, (32, 32))
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image = np.array(image)

#255 is white and 0 is black
images = [image, image]
images = np.array(images)
print(len(images))
images=np.append(images, np.array([image]), axis=0)
print(len(images))
cv.imwrite("test.png", image)