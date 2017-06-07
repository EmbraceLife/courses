# Goals:


#############################################
# save large arrays using bcolz

import bcolz
# create a folder and save data inside
def bz_save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def bz_load_array(fname):
    return bcolz.open(fname)[:]

# save_array(trained_model_path+"/preds_bc", preds)
# preds_bc = load_array(trained_model_path+"/preds_bc")


#############################################
# convert batch_iterators into large arrays

from tensorflow.contrib.keras.python.keras.preprocessing.image import DirectoryIterator
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator

get_batches_from_dir = DirectoryIterator
func_imageDataGenerator = ImageDataGenerator

import numpy as np
# turn DirectoryIterator into arrays
def iterator2array(data_dir):
	# turn iterators into arrays
    batch_iterator = get_batches_from_dir(
            directory = data_dir,
            image_data_generator=func_imageDataGenerator(),
            target_size=(224, 224),
            color_mode = "rgb", # add up to (224,224,3)
			#    classes=["dogs", "cats"],
            # class_mode=None, # no label is included
            # class_mode='binary', # label 1D is included
            class_mode='categorical', # label 2D is included, one-hot encoding included, i think;
            batch_size=1,
            shuffle=False, # so that images and labels order can be matched
            seed=123,
            data_format="channels_last")


    img_array = np.concatenate([batch_iterator.next()[0] for i in range(batch_iterator.samples)])
    lab_array = np.concatenate([batch_iterator.next()[1] for i in range(batch_iterator.samples)])

    return img_array, lab_array

#############################################
# # save arrays using numpy
#
# # save preds for decoding
# import numpy as np
# def np_save(dir_path, large_array):
#     np.save(dir_path, large_array)
#
# # load preds.npy
# def np_load(npy_file_path):
#     return np.load(npy_file_path)


#############################################
# # save arrays or objects in pickle
# import pickle
#
# def pk_save(dir_path, large_array):
#     with open(dir_path+".pickle", "wb") as f:
#         pickle.dump(large_array, f)
#
# def pk_load(dir_path):
#     with open(dir_path+".pickle", "rb") as f:
#         large_array = pickle.load(f)
#     return large_array


# ######################
# # save and load with torch.save, torch.load
# import torch
# torch.save(train_img_array, trained_model_path+"/train_img_array_torch")
# train_img_array_torch = torch.load(trained_model_path+"/train_img_array_torch")


#############################################
# # save and load with kur.idx
# from kur.utils import idx
#
# def idx_save(dir_path, large_array):
#     idx.save(dir_path, large_array)
#
# def idx_load(dir_path):
#     return idx.load(dir_path)
