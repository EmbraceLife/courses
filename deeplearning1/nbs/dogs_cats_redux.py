#Create references to important directories we will use over and over
import os, sys
current_dir = os.getcwd()
data_path = '/Users/Natsume/Downloads/data_for_all/dogscats'
LESSON_HOME_DIR = current_dir
DATA_HOME_DIR = data_path  #current_dir+'/data/redux'

#Allow relative imports to directories above lesson1/
sys.path.insert(1, os.path.join(sys.path[0], '..'))

#import modules
from utils import *
from vgg16 import Vgg16


# %cd $DATA_HOME_DIR

#Set path to sample/ path if desired
path = DATA_HOME_DIR + '/sample/' # '/'
test_path = DATA_HOME_DIR + '/test/' #We use all the test data
results_path=DATA_HOME_DIR + '/results/'
train_path=path + '/train/'
valid_path=path + '/valid/'

#import Vgg16 helper class
vgg = Vgg16()
