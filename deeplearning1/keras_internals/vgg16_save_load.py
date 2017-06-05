from vgg16_fit_fit_generator import vgg16

sample_test_path = "/Users/Natsume/Downloads/data_for_all/dogscats/sample/test"

####################
# test on model vgg16

# preds = vgg16.test()
#
# # save
# print('test before save: ', model.predict(X_test[0:2]))
# model.save('my_model.h5')   # HDF5 file, you have to pip3 install h5py if don't have it
# del model  # deletes the existing model
#
# # load
# model = load_model('my_model.h5')
# print('test after load: ', model.predict(X_test[0:2]))
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
