# All the required libraries are imported here
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers import Dropout
import csv
import os
import cv2
import sklearn
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
import matplotlib.pyplot as plt

#################################################
#Reading in the csv file containing the images and 
#the steering information

samples = []
with open('./driving_log.csv') as csvfile:
	reader = csv.reader (csvfile)
	for line in reader:
		samples.append (line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


#building the generator function to reduce the amount of data needed to store in memory
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle (X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)



#################################################

#Building the model for the CNN as given by NVIDIA
model = Sequential()
model.add (Lambda(lambda x: x /255.0 - 0.5, input_shape= (160, 320,3)))
model.add (Cropping2D(cropping=((70,25),(0,0))))
model.add (Convolution2D(24,5,5,subsample=(2,2),activation = "relu"))
model.add (Convolution2D(36,5,5,subsample=(2,2),activation = "relu"))
model.add (Convolution2D(48,5,5,subsample=(2,2),activation = "relu"))
model.add (Convolution2D(64,3,3,activation = "relu"))
model.add (Convolution2D(64,3,3,activation = "relu"))
model.add (Flatten())
model.add (Dense(100))
model.add(Dropout (0.3))
model.add (Dense(50))
model.add (Dense(10))
model.add(Dropout (0.3))
model.add (Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= 
	             len(train_samples), validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)


model.save('model.h5')
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()




