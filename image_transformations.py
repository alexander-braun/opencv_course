import cv2 as cv
import numpy as np

img = cv.imread('./Resources/Photos/park.jpg')

#cv.imshow('Park', img)


# translation
def translate(img, x, y):
  translateMatrix = np.float32([[1, 0, x], [0, 1, y]])
  dimensions = (img.shape[1], img.shape[0])
  return cv.warpAffine(img, translateMatrix, dimensions)

translated = translate(img, -100, 100)
#cv.imshow('Translated', translated)

# rotation
def rotate(img, angle, rotationPoint = None):
  (height, width) = img.shape[:2]
  if rotationPoint == None:
    rotationPoint = (width//2, height//2)
    
  rotationMatrix = cv.getRotationMatrix2D(rotationPoint, angle, 1.0)
  dimensions = (width, height)
  return cv.warpAffine(img, rotationMatrix, dimensions)
  
rotated = rotate(img, -45)
#cv.imshow('Rotated', rotated)

# resize
resized = cv.resize(img, (400, 400), interpolation=cv.INTER_CUBIC)
#cv.imshow('Resized', resized)

# flip
flipped = cv.flip(img, -1) # 0, 1, -1
#cv.imshow('Flipped', flipped)

# crop
cropped = img[200:400, 300:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)