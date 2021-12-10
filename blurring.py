import cv2 as cv
import numpy as np

img = cv.imread('./Resources/Photos/cats.jpg')
cv.imshow('Img', img)

# Averaging -> finds average of surrounding pixels
avg = cv.blur(img, (7, 7))
cv.imshow('Average', avg)

# Gaussian Blur -> each surrounding pixel gets weight, products of weights / surrounding pixels = result
gauss = cv.GaussianBlur(img, (7, 7), 0)
cv.imshow('Gaussian', gauss)

# Median Blur -> finds medium of surrounding pixels ( good for noise reduction )
median = cv.medianBlur(img, 7)
cv.imshow('Median', median)

# Bilatteral Blur -> weird, google it
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('Bilateral', bilateral)

cv.waitKey(0)