import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from methods import pre
from methods import train_test_transform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.layers import Dense, Activation, Dropout
from keras.activations import relu, sigmoid
from tensorflow import keras


def model_acc(X_train, X_test, y_train, y_test):
    tf.convert_to_tensor(X_train,dtype=float)
    tf.convert_to_tensor(y_train)
    model = Sequential()
    model.add(Dense(80, input_dim=38))
    model.add(Dense(40))
    model.add(Dropout(0.3,input_shape=(None,30)))
    model.add(Dense(20))
    model.add(Dense(1, activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(X_train, y_train, validation_split=0.3, epochs=1000, batch_size=64, verbose=0)

    predictions = model.predict(X_test) > 0.5

    _, accuracy = model.evaluate(X_test, y_test)
    
    return accuracy*100

def model1_acc(X_train, X_test, y_train, y_test):
    tf.convert_to_tensor(X_train,dtype=float)
    tf.convert_to_tensor(y_train)
    model = Sequential()
    model.add(Dense(80, input_dim=36))
    model.add(Dense(40))
    model.add(Dropout(0.3,input_shape=(None,30)))
    model.add(Dense(20))
    model.add(Dense(1, activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(X_train, y_train, validation_split=0.3, epochs=1000, batch_size=64, verbose=0)
   
    predictions = model.predict(X_test) > 0.5

    _, accuracy = model.evaluate(X_test, y_test)
    
    return accuracy*100