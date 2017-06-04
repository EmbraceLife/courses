###############################
# get sample dataset ready
import os, sys
current_dir = os.getcwd()
data_path = '/Users/Natsume/Downloads/data_for_all/dogscats'
LESSON_HOME_DIR = current_dir
DATA_HOME_DIR = data_path  #current_dir+'/data/redux'

#Set path to sample/ path if desired
path = DATA_HOME_DIR + '/sample' # '/'
test_path = DATA_HOME_DIR + '/test/' #We use all the test data
results_path=DATA_HOME_DIR + '/results/'
train_path=path + '/train/'
valid_path=path + '/valid/'

###############################
## Experiment on the functionality of vgg16 module

#import modules
from utils import *
from vgg16 import Vgg16
import vgg16 as v

## Experiment on vgg16.vgg_preprocess
# What it is like to change rgb to bgr?
image = np.linspace(0, 255, 75).reshape(3, 5, 5)
img_prep = v.vgg_preprocess(image)

###############################
# when initialize Vgg16, first create vgg model skeleton
# Vgg16.get_classes() load classes index from an external file
vgg = Vgg16()
# model skeleton filled with imagenet weights by Vgg16.create()
print vgg.model.summary()
# check 1000 image classes in imagenet training set
# print vgg.class

#Set constants. You can experiment with no_of_epochs to improve the model
batch_size=1 # 32 # train 1 sample at a batch to speed up the log print
no_of_epochs=1


###############################
#Finetune the model
batches = vgg.get_batches(train_path, batch_size=batch_size)
val_batches = vgg.get_batches(valid_path, batch_size=batch_size/2)
## many attrs inside batches and val_batches
 # 'batch_index',
 # 'batch_size',
 # 'class_indices',
 # 'class_mode',
 # 'classes',
 # 'color_mode',
 # 'dim_ordering',
 # 'directory',
 # 'filenames',
 # 'image_data_generator',
 # 'image_shape',
 # 'index_generator',
 # 'lock',
 # 'nb_class',
 # 'nb_sample',
 # 'next',
 # 'reset',
 # 'save_format',
 # 'save_prefix',
 # 'save_to_dir',
 # 'shuffle',
 # 'target_size',
 # 'total_batches_seen'
batches.batch_size
# get a batch of images and labels from batches of sample/train
img_train_batch, label_train_batch = batches.next()
img_val_batch, label_val_batch = val_batches.next()


#################################
# remove the last layer of vgg, replace with a FC layer
# make all other layers non-trainable, but the last new layer
# replace 1000 classes with new classes (2, and put them in order) with batches.nb_classes
vgg.finetune(batches)

### set learning rate for optimizer
#Not sure if we set this for all fits
vgg.model.optimizer.lr = 0.01

#Notice we are passing in the validation dataset to the fit() method
#For each epoch we test our model against the validation set
latest_weights_filename = None
for epoch in range(no_of_epochs):
    print "Running epoch: %d" % epoch
    vgg.fit(batches, val_batches, nb_epoch=1)
    latest_weights_filename = 'ft%d.h5' % epoch
    vgg.model.save_weights(results_path+latest_weights_filename)
print "Completed %s fit operations" % no_of_epochs
