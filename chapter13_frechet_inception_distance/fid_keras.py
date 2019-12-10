import numpy as np
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize


def scale_images(images, new_shape):
	images_list = list()
	for img in images:
		new_img = resize(img, new_shape, 0)
		images_list.append(new_img)
	return np.asarray(images_list)

def calculate_fid(model, images1, images2):
	act1 = model.predict(images1)
	act2 = model.predict(images2)

	# cal mean and covariance matrix
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

	ssdiff = np.sum((mu1-mu2) ** 2.0)

	covmean = sqrtm(sigma1.dot(sigma2))

	if np.iscomplexobj(covmean):
		covmean = covmean.real

	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

	return fid

if __name__ == "__main__":
	model = InceptionV3(include_top=False, pooling="avg", input_shape=(299, 299, 3))

	images1 = np.random.randint(0, 255, 10*32*32*3)
	images1 = images1.reshape((10, 32, 32, 3))
	images2 = np.random.randint(0, 255, 10*32*32*3)
	images2 = images2.reshape((10, 32, 32, 3))

	print("Img1 shape: {}, img2 shape: {}".format(images1.shape, images2.shape))

	images1 = images1.astype("float32")
	images2 = images2.astype("float32")

	images1 = scale_images(images1, (299, 299, 3))

	images2 = scale_images(images2, (299, 299, 3))

	print("Scaled image1 shape: {}, image2 shape: {}".format(images1.shape, images2.shape))

	# preprocess image
	images1 = preprocess_input(images1)
	images2 = preprocess_input(images2)

	# should equal to 0
	fid = calculate_fid(model, images1, images1)
	print("FID score (same): {:.3f}".format(fid))

	fid = calculate_fid(model, images1, images2)
	print("FID Score (different): {:.3f}".format(fid))