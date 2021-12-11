import os
import cv2 as cv
import numpy as np

# Get folder names
people = []
for i in os.listdir('./Resources/Faces/train'):
  people.append(i)
  
DIR = r'./Resources/Faces/train'

haar_cascade = cv.CascadeClassifier('./haar_face.xml')

features = []
labels = []

def create_train():
  for person in people:
    path = os.path.join(DIR, person)
    label = people.index(person)
    
    for img in os.listdir(path):
      img_path = os.path.join(path, img)
      img_array = cv.imread(img_path)
      gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
      face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
      
      for (x, y, w, h) in face_rect:
        faces_region_of_interest = gray[y:y+h, x:x+w]
        features.append(faces_region_of_interest)
        labels.append(label)
        
create_train()
print('Training done-------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# train recognizer on the features list and labels list
face_recognizer.train(features, labels)
face_recognizer.save('face_trained.yml')


