import cv2 as cv
import numpy as np

img = cv.imread('./Resources/Photos/park.jpg')
cv.imshow('Img', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

b,g,r = cv.split(img)
cv.imshow('B', b)
cv.imshow('G', g)
cv.imshow('R', r)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)


# merge color channels
merged = cv.merge([b, g, r])
cv.imshow('Merged', merged)

# show channels
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])
cv.imshow('blue', blue)
cv.imshow('green', green)
cv.imshow('red', red)

cv.waitKey(0)