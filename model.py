import cv2
import csv
import numpy as np
import os
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def readCSVFile():
    lines = []
    with open('data' + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        for line in reader:
            lines.append(line)
    return lines

# Image's path are converted to project specific paths
# Please note following path are supported for windwos OS
def updateFilePath(path):
    returnPath = path.strip()
    returnPath = "data\\IMG\\" + returnPath.split('\\')[-1]
    return returnPath

def getImagePathsAndAngles():
    center = []
    left = []
    right = []
    measurements = []
    lines = readCSVFile()
    for line in lines:
        measurements.append(float(line[3]))
        center.append(updateFilePath(line[0]))
        left.append(updateFilePath(line[1]))
        right.append(updateFilePath(line[2]))
    
    correction = 0.2
    paths = []
    paths.extend(center)
    paths.extend(left)
    paths.extend(right)
    angles = []
    angles.extend(measurements)
    angles.extend([x + correction for x in measurements])
    angles.extend([x - correction for x in measurements])
    return (paths, angles)

#Create generator to reduce in-memory overhead
#This Generator will load the images on the fly 
#This will also add additional data by flipping the images
def getGenerator(samples, batch_size=32):
    num_samples = len(samples)
    while True: 
        # Randomize the data before we process
        samples = sklearn.utils.shuffle(samples)
        done = True
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for path, angle in batch_samples:
                # load image
                originalImage = cv2.imread(path)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(angle)
                
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(angle*-1.0)

            # convert to np array
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

# Reading images and data.
paths, angles = getImagePathsAndAngles()
print('Total Images: {}'.format( len(paths)))

# combining paths and angles 
samples = list(zip(paths, angles))

# Split 80% data
train_samples, valid_samples = train_test_split(samples, test_size=0.2)

print('Train samples count: {}'.format(len(train_samples)))
print('Validation samples count: {}'.format(len(valid_samples)))

# Get generators for train and validation
train_generator = getGenerator(train_samples, batch_size=32)
valid_generator = getGenerator(valid_samples, batch_size=32)


# using Nvidia Self-driving Car CNN
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))

model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Compiling and training the model
model.compile(optimizer='adam', loss='mse')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_samples), validation_data=valid_generator, \
                 nb_val_samples=len(valid_samples), nb_epoch=3, verbose=1)

model.save('model.h5')
print("Model Saved")