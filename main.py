import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

mnist = np.load('mnist.npz')
(x_train, y_train) = mnist['x_train'], mnist['y_train']
(x_test, y_test) = mnist['x_test'], mnist['y_test']

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.save('handwritten.keras')

