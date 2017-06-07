################################
# Goal:
# 1. create train_batches, val_batches, test_batches from train folder, validation folder, and test folder for dogs and cats images
# 2. train and validation folder have sub-directories on cats and dogs separately, test folder has a single unknown subfolder for images with just index on its filename
# 2. the three batches above are objects of DirectoryIterator class


# among all save and load functions from numpy, pickle, torch, kur.utils.idx, bcolz can shrink large array 4 times smaller

from save_load_large_array import bz_save_array, bz_load_array, np_save, np_load, pk_save, pk_load, idx_save, idx_load, iterator2array, get_batches_from_dir, func_imageDataGenerator

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



trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"

# convert batch_iterators into arrays
train_img_array, train_lab_array = iterator2array(sample_train_path)
val_img_array, val_lab_array = iterator2array(sample_val_path)
test_img_array, test_lab_array = iterator2array(sample_test_path)

################################################
# save and load each array using bcolz
bz_save_array(trained_model_path+"/train_img_array", train_img_array)
bz_save_array(trained_model_path+"/train_lab_array", train_lab_array)
bz_save_array(trained_model_path+"/val_img_array", val_img_array)
bz_save_array(trained_model_path+"/val_lab_array", val_lab_array)
bz_save_array(trained_model_path+"/test_img_array", test_img_array)
bz_save_array(trained_model_path+"/test_lab_array", test_lab_array)

# load the check the arrays
try_img_array = bz_load_array(trained_model_path+"/train_img_array")
try_lab_array = bz_load_array(trained_model_path+"/train_lab_array")


################################################
# # save and load array using np_save, np_load
# np_save(trained_model_path+"/train_img_array", train_img_array)
# try_img_array = np_load(trained_model_path+"/train_img_array.npy")


######################
# save and load with pickle
# pk_save(trained_model_path+"/train_img_array_pk", train_img_array)
# try_img_array = pk_load(trained_model_path+"/train_img_array_pk")


######################
# idx_save(trained_model_path+"/train_img_array_idx", train_img_array)
# train_img_idx = idx_load(trained_model_path+"/train_img_array_idx")
