from vgg16_iterator_from_directory import train_batches, val_batches
from vgg16_finetune import vgg16



## check model summary
vgg16.summary()



vgg16.fit_generator(
					generator=train_batches,
					steps_per_epoch=1,
					epochs=2,
					verbose=2,
					callbacks=None,
					validation_data=val_batches,
					validation_steps=1,
					class_weight=None,
					max_q_size=10,
					workers=1,
					pickle_safe=False,
					initial_epoch=0)


#Notice we are passing in the validation dataset to the fit() method
#For each epoch we test our model against the validation set
# latest_weights_filename = None
# for epoch in range(no_of_epochs):
#     print "Running epoch: %d" % epoch
#     vgg.fit(batches, val_batches, nb_epoch=1)
#     latest_weights_filename = 'ft%d.h5' % epoch
#     vgg.model.save_weights(results_path+latest_weights_filename)
# print "Completed %s fit operations" % no_of_epochs



############################################
# ## create fake data for training
# # input_1 (InputLayer)         (None, 224, 224, 3)
# fake_img = np.random.random((32*10, 224, 224, 3))
# fake_lab = np.random.random((32*10, 1000))
#
# lr = 0.001
# vgg16.compile(optimizer=Adam(lr=lr),
# 		loss='categorical_crossentropy', metrics=['accuracy'])
#
# vgg16.fit(x=fake_img, y=fake_lab, batch_size=32, epochs=1, verbose=2, callbacks=None, validation_split=0.1, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
