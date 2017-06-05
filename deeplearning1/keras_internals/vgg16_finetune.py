import tensorflow as tf
import numpy as np


def ft(vgg16, num):
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
	model = vgg16.model
	# layers inside model is stored in a list
	# remove the last layer from the model
	model.pop()
	# make all layers non-trainable
	for layer in model.layers: layer.trainable=False
	# add a new Dense layer to model, with num neurons, and softmax activations
	model.add(Dense(num, activation='softmax'))
	# recompile this new model
	vgg16.compile()


def finetune(vgg16, batches, num_classes):
	"""
		Modifies the original VGG16 network architecture and updates vgg16.classes for new training data.

		Args:
			batches : A keras.preprocessing.image.ImageDataGenerator object.
					  See definition for get_batches().
	"""
	# last new layer's neuron number determined by number of classes
	ft(vgg16, num_classes)

	classes = list(iter(batches.class_indices)) # get a list of all the class labels

	# batches.class_indices is a dict with the class name as key and an index as value
	# eg. {'cats': 0, 'dogs': 1}

	# sort the class labels by index according to batches.class_indices and update model.classes
	for c in batches.class_indices:
		classes[batches.class_indices[c]] = c
	vgg16.classes = classes


def compile(vgg16, lr=0.001):
	"""
		Configures the model for training.
		See Keras documentation: https://keras.io/models/model/
	"""
	vgg16.model.compile(optimizer=Adam(lr=lr),
			loss='categorical_crossentropy', metrics=['accuracy'])


# image batch dataset are processed in tf.contrib.keras.preprocessing.image

# DirectoryIterator(Iterator):\n',
#   '  """Iterator capable of reading images from a directory on disk.\n',

# class ImageDataGenerator(object):\n',
#   '  """Generate minibatches of image data with real-time data augmentation.\n',
