# copyright (c) 2022 Aaron Hance
# slither_data.js is freely distributable under the MIT license.
# http://www.opensource.org/licenses/mit-license.php
    

from cgi import test
import numpy as np
import json
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from sqlalchemy import false
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait as WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
from concurrent.futures import thread
from itertools import count
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import initializers
import random
import os
import gc
import cv2
import pickle

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


class DQN:
    #learning rate
    lr = 0.001
    batch_size = 32
    model = None
    optimizer = keras.optimizers.Adam(lr=lr)
    loss_func = keras.losses.mean_squared_error
    state_size = (160, 160, 3)
    action_size = 4


    def __init__(self):
        #build model
        self.model = self.build_model()
        #compile model
        self.model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=self.lr), metrics=['accuracy'])
        #load model if exists

        #print model summary
        self.model.summary()

    #build model
    def build_model(self):
        #model
        model = keras.Sequential()
        #convolutional layers
        model.add(keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=self.state_size))
        model.add(keras.layers.Conv2D(16, (5, 5), activation='relu', input_shape=self.state_size))
        model.add(keras.layers.Conv2D(16, (7, 7), activation='relu', input_shape=self.state_size))
        #max pooling layer
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        #flatten layer
        model.add(keras.layers.Flatten())
        #dense layer
        model.add(keras.layers.Dense(16, activation='relu'))
        #output layer
        model.add(keras.layers.Dense(self.action_size, activation='softmax'))
        #return model
        return model


data = pickle.load(open("samples.pickle", "rb"))

data = np.array(data)

x = np.array([i['screen'] for i in data])
y = np.array([i['keys'] for i in data])

print(x[0].shape)
print(x[-1].shape)
print(y[0].shape)
print(y[-1].shape)

print(y[:20])

dqn = DQN()

model = dqn.model

model.fit(x, y, batch_size=dqn.batch_size, epochs=1, verbose=1, validation_split=0.1)

#save model weights
model.save_weights('v3model.h5')