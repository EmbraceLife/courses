import tensorflow as tf
import numpy as np
from tensorflow.contrib.keras.python.keras.models import Model

from vgg16_tf_kr import vgg16

Dense = tf.contrib.keras.layers.Dense
Adam = tf.contrib.keras.optimizers.Adam

def ft(vgg16, num, lr = 0.001):
	"""
		Replace the last layer of the model with a Dense (fully connected) layer of num neurons.
		Will also lock the weights of all layers except the new layer so that we only learn
		weights for the last layer in subsequent training.

		Args:
			num (int) : Number of neurons in the Dense layer
		Returns:
			None
	"""
	# access the vgg16 model
	# model = vgg16.model
	# layers inside model is stored in a list
	# remove the last layer from the model
	vgg16.layers.pop()
	# make all layers non-trainable
	for layer in vgg16.layers: layer.trainable=False
	# add a new Dense layer to model, with num neurons, and softmax activations
	x = Dense(num, activation='softmax', name='predictions')(vgg16.layers[-1].output)
	# recompile this new model
	vgg16 = Model(inputs=vgg16.input, outputs=x)
	vgg16.compile(optimizer=Adam(lr=lr),
			loss='categorical_crossentropy', metrics=['accuracy'])

	return vgg16
						 # set num=1, class_mode='binary'
vgg16 = ft(vgg16, num=2) # if num = 2, class_mode="categorical"

    # def finetune(vgg16, batches):
    #     """
    #         Modifies the original VGG16 network architecture and updates self.classes for new training data.
	#
    #         Args:
    #             batches : A keras.preprocessing.image.ImageDataGenerator object.
    #                       See definition for get_batches().
    #     """
    #     ft(vgg16, batches.num_class)
    #     classes = list(iter(batches.class_indices)) # get a list of all the class labels
	#
    #     # batches.class_indices is a dict with the class name as key and an index as value
    #     # eg. {'cats': 0, 'dogs': 1}
	#
    #     # sort the class labels by index according to batches.class_indices and update model.classes
    #     for c in batches.class_indices:
    #         classes[batches.class_indices[c]] = c
    #     self.classes = classes


# def compile(self, lr=0.001):
#     """
#         Configures the model for training.
#         See Keras documentation: https://keras.io/models/model/
#     """
#     self.model.compile(optimizer=Adam(lr=lr),
#             loss='categorical_crossentropy', metrics=['accuracy'])
