import cv2 as cv
import numpy as np

img = cv.imread('./Resources/Photos/cats.jpg')

blank = np.zeros(img.shape, dtype='uint8')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny', canny)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(len(contours))

cv.drawContours(blank, contours, -1, (0, 0, 255), 1)

cv.imshow('Img Contours', blank)

cv.waitKey(0)