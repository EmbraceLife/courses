################################
# Goal:
# 1. create train_batches, val_batches, test_batches from train folder, validation folder, and test folder for dogs and cats images
# 2. train and validation folder have sub-directories on cats and dogs separately, test folder has a single unknown subfolder for images with just index on its filename
# 2. the three batches above are objects of DirectoryIterator class



# method1
from tensorflow.contrib.keras.python.keras.preprocessing.image import DirectoryIterator
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator

get_batches_from_dir = DirectoryIterator
func_imageDataGenerator = ImageDataGenerator

# # method 2:
# get_batches_from_dir = tf.contrib.keras.preprocessing.image.DirectoryIterator
# func_imageDataGenerator = tf.contrib.keras.preprocessing.image.ImageDataGenerator

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
img, lab = train_batches.next()

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

sample_test_path = "/Users/Natsume/Downloads/data_for_all/dogscats/sample/test"
test_batches = get_batches_from_dir(directory = sample_test_path,
							   image_data_generator=func_imageDataGenerator(),
							   target_size=(224, 224),
							   color_mode = "rgb", # add up to (224,224,3)
							#    classes=["dogs", "cats"],
							   class_mode=None, # unknown about classes
							   batch_size=32,
							   shuffle=True,
							   seed=123,
							   data_format="channels_last"
							#    save_to_dir
							#    save_prefix
							#    save_format
							   )
