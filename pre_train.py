# copyright (c) 2022 Aaron Hance
# slither_data.js is freely distributable under the MIT license.
# http://www.opensource.org/licenses/mit-license.php
    

import numpy as np
import json
import time
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import initializers
import random
import os
import gc
import pickle
import keras_tuner as kt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  #build model
def build_model(hp):
    hp_filter_count_1 = hp.Choice('filter_count', values=[1, 2, 4, 5, 7, 10, 14, 16, 20, 25])
    hp_kernel_size_1 = hp.Choice('kernel_size', values=[3, 5, 7, 9, 11, 13, 15])

    hp_filter_count_2 = hp.Choice('filter_count_2', values=[1, 2, 4, 5, 7, 10, 14, 16, 20, 25])
    hp_kernel_size_2 = hp.Choice('kernel_size_2', values=[3, 5, 7, 9, 11, 13, 15])

    hp_filter_count_3 = hp.Choice('filter_count_3', values=[1, 2, 4, 5, 7, 10, 14, 16, 20, 25])
    hp_kernel_size_3 = hp.Choice('kernel_size_3', values=[3, 5, 7, 9, 11, 13, 15])

    #model
    model = keras.Sequential()
    #convolutional layers
    model.add(keras.layers.Conv2D(hp_filter_count_1, (hp_kernel_size_1, hp_kernel_size_1), activation='relu', input_shape=(160, 160, 3)))
    model.add(keras.layers.Conv2D(hp_filter_count_2, (hp_kernel_size_2, hp_kernel_size_2), activation='relu'))
    model.add(keras.layers.Conv2D(hp_filter_count_3, (hp_kernel_size_3, hp_kernel_size_3), activation='relu'))
    #max pooling layer
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #flatten layer
    model.add(keras.layers.Flatten())
    #dense layer
    model.add(keras.layers.Dense(16, activation='relu'))
    #output layer
    model.add(keras.layers.Dense(4, activation='softmax'))
    #return model
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return model


data = pickle.load(open("samples.pickle", "rb"))

data = np.array(data)
data = data[:int(len(data)/2)]

x = np.array([i['screen'] for i in data], dtype=np.float32)
y = np.array([i['keys'] for i in data], dtype=np.float32)

data = []
gc.collect()

print(x[0].shape)
print(x[-1].shape)
print(y[0].shape)
print(y[-1].shape)

print(y[520:590])

tuner = kt.Hyperband(build_model, max_epochs=2, factor=3, objective='val_accuracy')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(x, y, epochs=2, validation_split=0.1, callbacks=[stop_early])

print(tuner.results_summary())

best_model = tuner.get_best_models(1)[0]

# model = dqn.model
# model.fit(x, y, batch_size=dqn.batch_size, epochs=1, verbose=1, validation_split=0.1)

#save model weights
#model.save_weights('v3model.h5')