import cv2 as cv
import numpy as np
img = cv.imread('find and draw contours in open CV/none.jpg', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (512, 512))
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 20, 255, 0)

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print('Number of contours:', str(len(contours)))
print(contours[0])


cv.drawContours(img, contours, -1, (0, 255, 0), 3)
cv.imshow('Image', img)
cv.imshow('Image Gray', imgray)
cv.waitKey(0)
cv.destroyAllWindows()
