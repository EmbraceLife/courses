# test folder images are made into batches
from vgg16_iterator_from_directory import test_batches
import tensorflow as tf

#############################################
load_model = tf.contrib.keras.models.load_model
# load re-trained vgg16 model to test
trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"
vgg16 = load_model(trained_model_path+'/tfkr_vgg16.h5') # already compiled


#############################################
# test model with test_batches
preds = vgg16.predict_generator(generator=test_batches,
								steps=1,
								max_q_size=10,
								workers=1,
								pickle_safe=False,
								verbose=2
								)

print(preds)

# save preds for decoding
import numpy as np
np.save(trained_model_path+"/preds", preds)

# load preds.npy
trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"
preds = np.load(trained_model_path+"/preds.npy")
