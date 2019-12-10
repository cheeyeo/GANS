from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.datasets import cifar10
import numpy as np
import math
from skimage.transform import resize

def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return np.asarray(images_list)

def calculate_inception_score(images, n_split=10, eps=1E-16):
	model = InceptionV3()

	scores = list()
	n_part = math.floor(images.shape[0] / n_split)

	for i in range(n_split):
		# get p(y|x)
		ix_start, ix_end = i * n_part, (i+1)*n_part
		subset = images[ix_start:ix_end]
		subset = subset.astype("float32")
		subset = scale_images(subset, (299, 299, 3))

		# preprocess images, scale to [-1, 1]
		subset = preprocess_input(subset)

		# calculate p(y|x)
		p_yx = model.predict(subset)
		# calculate p(y)
		p_y = np.expand_dims(p_yx.mean(axis=0), 0)

		# calculate KL divergence using log probs
		kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))

		# sum over classes
		sum_kl_d = kl_d.sum(axis=1)

		# average over images
		avg_kl_d = np.mean(sum_kl_d)

		is_score = np.exp(avg_kl_d)

		scores.append(is_score)

	is_avg = np.mean(scores)
	is_std = np.std(scores)

	return is_avg, is_std


if __name__ == "__main__":
	# Test set with all ones as probs
	# Will cause IS score to be 1.0 due to the uniform distribution across all predicted probs....
	# imgs = np.ones((50, 299, 299, 3))
	# print("Images: {}".format(imgs.shape))

	# is_avg, is_std = calculate_inception_score(imgs, )

	# print("Scores: {}, {}".format(is_avg, is_std))

	# Test using the CIFAR-10 dataset
	(images, _), (_, _) = cifar10.load_data()
	np.random.shuffle(images)

	print("Images: {}".format(images.shape))

	is_avg, is_std = calculate_inception_score(images)
	print("Scores: {}, {}".format(is_avg, is_std))