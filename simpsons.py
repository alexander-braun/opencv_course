import os
import caer
import canaro
import numpy as np
import cv2 as cv
import gc
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler

# Dataset here: https://www.kaggle.com/alexattia/the-simpsons-characters-dataset

IMG_SIZE = (80, 80)
channels = 1
char_path = r'C:\Users\Alex\Desktop\Apps\Simpsons_Dataset\simpsons_dataset'

char_dict = {}
for char in os.listdir(char_path):
  # character name + length of folder into dictionary
  char_dict[char] = len(os.listdir(os.path.join(char_path, char)))
  
# sort dictionary
char_dict = caer.sort_dict(char_dict, descending=True)

# take first 10 characters
characters = []
count = 0
for i in char_dict:
  characters.append(i[0])
  count += 1
  if count >= 10:
    break
  
# create training data
train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)
# plt.figure(figsize=(30, 30))
# plt.imshow(train[0][0], cmap='gray')
# plt.show()

# seperate trainingset into features and labels
featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)

# Normalize featureSet to be in range (0, 1)
featureSet = caer.normalize(featureSet)
labels = to_categorical(labels, len(characters))

# Create training and validation data
# Model is training on training data and test itself on validation data
x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=.2)

# cleanup
del train
del featureSet
del labels
gc.collect()

# Create image data generator
BATCH_SIZE = 32
EPOCHS = 10

datagen = canaro.generators.imageDataGenerator()
training_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

# Create the model
model = canaro.models.createSimpsonsModel(
  IMG_SIZE=IMG_SIZE, 
  channels=channels, 
  output_dim=len(characters), 
  loss='binary_crossentropy', 
  decay=1e-6, 
  learning_rate=0.001, 
  momentum=.9, 
  nesterov=True
)

# create callbacks list with learning-rate-scheduler 
callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]

# Train the model
train = model.fit(
  training_gen, 
  steps_per_epoch=len(x_train) // BATCH_SIZE, 
  epochs=EPOCHS, 
  validation_data=(x_val, y_val), 
  validation_steps=len(y_val) // BATCH_SIZE, 
  callbacks=callbacks_list
)

# test how good our model is
test_path = r'C:\Users\Alex\Desktop\Apps\Simpsons_Dataset\kaggle_simpson_testset\kaggle_simpson_testset\bart_simpson_24.jpg'
img = cv.imread(test_path)

plt.imshow(img)
plt.show()

def prepare(img):
  img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  img = cv.resize(img, IMG_SIZE)
  img = caer.reshape(img, IMG_SIZE, 1)
  return img

predictions = model.predict(prepare(img))
print(characters[np.argmax(predictions[0])])
                 
    
