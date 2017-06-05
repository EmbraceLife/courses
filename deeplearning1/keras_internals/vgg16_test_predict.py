import 

def test(self, path, batch_size=8):
	"""
		Predicts the classes using the trained model on data yielded batch-by-batch.
		Args:
			path (string):  Path to the target directory. It should contain one subdirectory
							per class.
			batch_size (int): The number of images to be considered in each batch.

		Returns:
			test_batches, numpy array(s) of predictions for the test_batches.

	"""
	test_batches = self.get_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
	return test_batches, self.model.predict_generator(test_batches, test_batches.nb_sample)
