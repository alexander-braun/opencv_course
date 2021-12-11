import cv2 as cv
import numpy as np
import os

haar_cascade = cv.CascadeClassifier('./haar_face.xml')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('./face_trained.yml')

people = []
for i in os.listdir('./Resources/Faces/train'):
  people.append(i)
  
img = cv.imread(r'./Resources/Faces/val/gettyimages-861097674-2048x2048.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces_rect:
  faces_region_of_interest = gray[y:y+h, x:x+h]
  label, confidence = face_recognizer.predict(faces_region_of_interest)
  print(people[label], confidence)
  cv.putText(img, str(people[label]), (x, y), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
  cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
  
cv.imshow('Detected', img)
cv.waitKey(0)