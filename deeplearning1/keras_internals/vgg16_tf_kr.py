################################
### goal
# 1. load VGG16 model (create model and load its pretrained weights)
# 2. downloaded and saved locally at ~/.keras/models/
# 3. its classes are download and saved here too
# 4. examine its source code
# 5. vgg16 is available as local variable in the end

# https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/contrib/keras/python/keras/applications/vgg16.py


import tensorflow as tf

# method 1: to load VGG16
from tensorflow.contrib.keras.python.keras.applications.vgg16 import VGG16
vgg16 = VGG16()

# # method 2 to load VGG16
# vgg16 = tf.contrib.keras.applications.vgg16.VGG16()

## check model summary
vgg16.summary()
