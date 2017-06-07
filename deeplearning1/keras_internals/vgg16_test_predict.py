# test folder images are made into batches
from vgg16_iterator_from_directory import test_batches


#############################################
from tensorflow.contrib.keras.python.keras.models import load_model
# load re-trained vgg16 model to test
trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"
vgg16_2class = load_model(trained_model_path+'/vgg16_2class.h5') # already compiled


#############################################
# test model with test_batches
preds = vgg16_2class.predict_generator(generator=test_batches,
								steps=1,
								max_q_size=10,
								workers=1,
								pickle_safe=False,
								verbose=2
								)

print(preds)


#############################################
# save large arrays using bcolz
from save_load_large_array import bz_save_array, bz_load_array

bz_save_array(trained_model_path+"/preds_bz", preds)
preds_bc = bz_load_array(trained_model_path+"/preds_bz")
