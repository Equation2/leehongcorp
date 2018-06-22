
import numpy
#import cPickle as pickle
import pickle
import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
#model.add(Lambda(lambda x: K.tf.nn.softmax(x)))
K.set_image_dim_ordering('th')
import pickle, gzip, numpy


# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')

#pickle.load(f, encoding='latin1')
train_set, valid_set, test_set = pickle.load(f,encoding='iso-8859-1')

f.close()

# 튜플을 이렇게 둘로 나눌 수 있음
X_train, y_train = train_set
X_test, y_test = test_set


X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32') 

X_train = X_train / 255
X_test = X_test / 255 

y_train = np_utils.to_categorical(y_train) 
y_test = np_utils.to_categorical(y_test) 
num_classes = y_test.shape[1] 

seed=7
numpy.random.seed(seed)

def cnn_model():
    # create model
    model = Sequential()
    model.add(Convolution2D(30, kernel_size=5, strides=5, padding='valid', input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(127, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = cnn_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
