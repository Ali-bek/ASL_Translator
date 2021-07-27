import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
train_dir = "gesture/asl_alphabet/train/"
test_dir = "gesture/asl_alphabet/test/"

for classe in classes :
    path = os.path.join (train_dir, classe)
    for image in os.listdir (path):
        image_arr = cv2.imread (os.path.join(path,image),cv2.IMREAD_GRAYSCALE)
        plt.imshow(image_arr)
        plt.show ()
        break
    break

IMG_SIZE = 40
image_resize_arr = cv2.resize(image_arr,(IMG_SIZE,IMG_SIZE))

training_data = []

def create_training_data ():
    for classe in classes :
        
        path = train_dir + "/" + classe
        for img in os.listdir(path):
            img_array = cv2.imread (os.path.join(path,img), 0)
            gray_frame = cv2.GaussianBlur(img_array, (9, 9), 0)
            new_array = cv2.resize (gray_frame, (IMG_SIZE,IMG_SIZE))
            training_data.append ([new_array, int(classe)])

create_training_data()

import random

random.shuffle(training_data)

X_train = []
y_train = []

for features, label in training_data :
    X_train.append(features)
    y_train.append(label)
    

X_train = np.array (X_train).reshape(-1,IMG_SIZE, IMG_SIZE, 1)

testing_data = []

def create_testing_data ():
    for classe in classes :
        path = test_dir + "/" + classe
        for img in os.listdir(path):
            img_array = cv2.imread (os.path.join(path,img), 0)
            gray_frame = cv2.GaussianBlur(img_array, (9, 9), 0)
            new_array = cv2.resize (gray_frame, (IMG_SIZE,IMG_SIZE))
            testing_data.append ([new_array, int(classe)])
            
             
create_testing_data ()

random.shuffle(testing_data)

X_test = []
y_test = []

for features, label in testing_data :
    X_test.append(features)
    y_test.append(label)
    

X_test = np.array (X_test).reshape(-1,IMG_SIZE, IMG_SIZE, 1)

from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = X_train.reshape (X_train.shape[0], *(40,40,1))
X_test = X_test.reshape (X_test.shape[0], *(40,40,1))

print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)

import tensorflow as tf
generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                            rotation_range=10,
                                                            zoom_range=0.10,
                                                            width_shift_range=0.1,
                                                            height_shift_range=0.1,
                                                            shear_range=0.1,
                                                            horizontal_flip=False,
                                                            fill_mode="nearest")

X_train_flow = generator.flow(X_train, y_train, batch_size=32)
X_val_flow = generator.flow(X_test, y_test, batch_size=32)

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D,MaxPooling2D
import pickle
from tensorflow.keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping

model = Sequential()

model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (40,40,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(28, activation = "softmax"))


optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()

epochs = 20  # for better result increase the epochs
batch_size = 128

history = model.fit(X_train,y_train, batch_size=batch_size, epochs = epochs, verbose=1 , validation_data = (X_test,y_test))

scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

model.save("model_cnn_number.h5")