# train vgg16 on dogscats and make vgg16 available
from vgg16_fit_fit_generator import vgg16


trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"
vgg16.save(trained_model_path+'/tfkr_vgg16.h5')

del vgg16  # deletes the existing model
#
# # load
# model = load_model('my_model.h5') # already compiled

# """
# # save and load weights
# model.save_weights('my_model_weights.h5')
# model.load_weights('my_model_weights.h5')
#
# # save and load fresh network without trained weights
# from keras.models import model_from_json
# json_string = model.to_json()
# model = model_from_json(json_string)
# """
