################################
### internal creation process
# https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/contrib/keras/python/keras/applications/vgg16.py


import tensorflow as tf

vgg16 = tf.contrib.keras.applications.vgg16.VGG16()


################################
### vgg16 module
# dr tf.contrib.keras.applications.vgg16

### attrs and methods
# ['VGG16', # init func
#  '__builtins__',
#  '__cached__',
#  '__doc__',
#  '__file__',
#  '__loader__',
#  '__name__',
#  '__package__',
#  '__path__',
#  '__spec__',
#  'decode_predictions',
#  'preprocess_input']


##########################
### internal creation process

############
# step 1: lazy_loader the keras module
# /Users/Natsume/miniconda2/envs/dlnd-tf-lab/lib/python3.5/site-packages/tensorflow/python/util/lazy_loader.py(52)
#
#   52  ->   def __getattr__(self, item):
#   53         module = self._load()
#   54         return getattr(module, item)
 # return <module 'tensorflow.contrib.keras' from '/Users/Natsume/miniconda2/envs/dlnd-tf-lab/lib/python3.5/site-packages/tensorflow/contrib/keras/__init__.py'>

############
# step 2: create vgg16
## see logic in source code
 # https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/contrib/keras/python/keras/applications/vgg16.py

##########################
### VGG16 weights
# downloaded and saved locally at ~/.keras/models/
# filename: vgg16_weights_tf_dim_ordering_tf_kernels.h5
# filesize: 500+ mb


##########################
### attrs and methods
# vgg16 has 136 attrs and methods: vgg16.variables
# sequential has 123 attrs and methods
