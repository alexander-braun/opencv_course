import cv2 as cv
import numpy as np

# and, or, Xor, not
# operate in binary manner -> turned off when 0, on when 1

blank = np.zeros((400, 400), dtype='uint8')

rect = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)

cv.imshow('Rectangle', rect)
cv.imshow('Circle', circle)

# Bitwise AND -> common regions (intersection) is beeing returned
bitwise_and = cv.bitwise_and(rect, circle)
cv.imshow('Bitwise And', bitwise_and)

# Bitwise Or -> elem1 or elem2 are displayed == like addition - common regions
bitwise_or = cv.bitwise_or(rect, circle)
cv.imshow('Bitwise Or', bitwise_or)

# Bitwise Xor -> non intersecting regions == regions both don't have in common
bitwise_xor = cv.bitwise_xor(rect, circle)
cv.imshow('Bitwise Or', bitwise_xor)

# bitwise not
bitwise_not = cv.bitwise_not(circle, rect)
cv.imshow('Bitwise Not', bitwise_not)

cv.waitKey(0)