import cv2 as cv
import numpy as np

img = cv.imread('./Resources/Photos/park.jpg')


cv.imshow('Img', img)

cv.waitKey(0)