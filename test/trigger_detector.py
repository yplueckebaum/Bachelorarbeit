import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# Basic architecture
#tf.keras.layers.experimental.preprocessing.Resizing(input_shape=)
convolution =layers.Conv2D(kernel_size=(3,3),strides=(2,2),filters=16)
inputs = keras.Input(shape=(None,None,1))#target from paper is 160,160,3 todo: revert input shape
resize = tf.keras.layers.experimental.preprocessing.Resizing(height=160,width=160)(inputs)
# todo interpolation method?
x1 = layers.Conv2D(kernel_size=(3,3),strides=(2,2),filters=16)(resize)
pool1 = layers.GlobalMaxPooling2D()(x1)
x2 = layers.Conv2D(kernel_size=(3,3),strides=(2,2),filters=16)(x1)
pool2 = layers.GlobalMaxPooling2D()(x2)
x3 = layers.Conv2D(kernel_size=(3,3),strides=(2,2),filters=16)(x2)
pool3 = layers.GlobalMaxPooling2D()(x3)
x4 = layers.Conv2D(kernel_size=(3,3),strides=(2,2),filters=16)(x3)
pool4 = layers.GlobalMaxPooling2D()(x4)
concat = layers.concatenate([pool1,pool2,pool3,pool4])
outputs = layers.Dense(1)(concat)
squished = tf.keras.layers.Activation(activation='sigmoid')(outputs)
model = keras.Model(inputs=inputs,outputs=squished)

#add optimizer
optimizer = keras.optimizers.Adam()
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])
#compile model
model.save('trigger_detector_architecture')



print(keras.utils.plot_model(model, "my_first_model.png",show_shapes=True))


