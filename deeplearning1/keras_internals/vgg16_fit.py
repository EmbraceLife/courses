import numpy as np
import tensorflow as tf
# from keras.optimizers import SGD, RMSprop, Adamb 15
Adam = tf.contrib.keras.optimizers.Adam
vgg16 = tf.contrib.keras.applications.vgg16.VGG16()

## check model summary
vgg16.summary()

## create fake data for training
# input_1 (InputLayer)         (None, 224, 224, 3)
fake_img = np.random.random((32*10, 224, 224, 3))
fake_lab = np.random.random((32*10, 1000))

lr = 0.001
vgg16.compile(optimizer=Adam(lr=lr),
		loss='categorical_crossentropy', metrics=['accuracy'])

vgg16.fit(fake_img, fake_lab, batch_size=32, epochs=1, verbose=2, callbacks=None, validation_split=0.1, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
