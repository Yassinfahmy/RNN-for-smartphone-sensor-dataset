# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 17:05:34 2021

@author: GuestUser
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve,percision_score, recall_score

import pandas as pd


# import testing and training sets
x_train         = pd.read_csv('C:/Users/GuestUser/Desktop/spring 2021/BME 578/Week 7 & 8/UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt',header=None, delim_whitespace=True)
y_train_uncoded = pd.read_csv('C:/Users/GuestUser/Desktop/spring 2021/BME 578/Week 7 & 8/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt',header=None, delim_whitespace=True)
x_test          = pd.read_csv('C:/Users/GuestUser/Desktop/spring 2021/BME 578/Week 7 & 8/UCI HAR Dataset/UCI HAR Dataset/test/X_test.txt',header=None, delim_whitespace=True)
y_test_uncoded  = pd.read_csv('C:/Users/GuestUser/Desktop/spring 2021/BME 578/Week 7 & 8/UCI HAR Dataset/UCI HAR Dataset/test/y_test.txt',header=None, delim_whitespace=True)

#create dummy variables of labels
y_train= pd.get_dummies(y_train_uncoded[0])
y_test = pd.get_dummies(y_test_uncoded[0])

#convert to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test  = np.array(x_test)
y_test  = np.array(y_test)

time_step=1
data_dim=561

# reshape input to be [samples, time steps, features]
x_train = np.reshape(x_train, (-1, time_step, x_train.shape[1]))
x_test = np.reshape(x_test, (-1, time_step, x_test.shape[1]))

#build a sequential model
model = Sequential()
model.add(LSTM(300, input_shape=(time_step, data_dim)))
model.add(Dense(6,activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15, batch_size=50)

# predict the testing dataset
y_predicted= model.predict(x_test)
y_predicted=np.around(y_predicted,0)


