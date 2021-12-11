import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('./Resources/Photos/group 1.jpg')
cv.imshow('Img', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

# Sobel
sobX = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobY = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined = cv.bitwise_or(sobX, sobY)
combined = np.uint8(np.absolute(combined))
cv.imshow("Combined", combined)


# Canny
canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)
  
cv.waitKey(0)