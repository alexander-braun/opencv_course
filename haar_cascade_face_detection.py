import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('./Resources/Photos/group 1.jpg')
# cv.imshow('Img', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

haar_cascade = cv.CascadeClassifier('./haar_face.xml')
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

print("Found", len(faces_rect), "faces")

for (x, y, w, h) in faces_rect:
  cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
  
cv.imshow('Detected', img)

cv.waitKey(0)