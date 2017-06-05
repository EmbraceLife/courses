import numpy as np
import tensorflow as tf

get_batches_from_dir = tf.contrib.keras.preprocessing.image.DirectoryIterator

func_imageDataGenerator = tf.contrib.keras.preprocessing.image.ImageDataGenerator

sample_train_path = "/Users/Natsume/Downloads/data_for_all/dogscats/sample/train"
train_batches = get_batches_from_dir(directory = sample_train_path,
							   image_data_generator=func_imageDataGenerator(),
							   target_size=(224, 224),
							   color_mode = "rgb", # add up to (224,224,3)
							#    classes=["dogs", "cats"],
							   class_mode="categorical", # binary for only 2 classes
							   batch_size=32,
							   shuffle=True,
							   seed=123,
							   data_format="channels_last"
							#    save_to_dir
							#    save_prefix
							#    save_format
							   )

sample_val_path = "/Users/Natsume/Downloads/data_for_all/dogscats/sample/valid"
val_batches = get_batches_from_dir(directory = sample_val_path,
							   image_data_generator=func_imageDataGenerator(),
							   target_size=(224, 224),
							   color_mode = "rgb", # add up to (224,224,3)
							#    classes=["dogs", "cats"],
							   class_mode="categorical", # for only 2 classes
							   batch_size=32,
							   shuffle=True,
							   seed=123,
							   data_format="channels_last"
							#    save_to_dir
							#    save_prefix
							#    save_format
							   )
