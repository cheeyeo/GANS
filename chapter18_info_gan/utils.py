import numpy as np
import math
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical

def generate_latent_points(latent_dim, cat, samples):
	z_latent = np.random.randn(latent_dim * samples)
	z_latent = z_latent.reshape(samples, latent_dim)

	cat_codes = np.random.randint(0, cat, samples)
	cat_codes = to_categorical(cat_codes, num_classes=cat)

	z_input = np.hstack((z_latent, cat_codes))

	return [z_input, cat_codes]

# Allows a passed digit to be set as category code
def generate_latent_points_with_digit(latent_dim, cat, samples, digit):
	z_latent = np.random.randn(latent_dim * samples)
	z_latent = z_latent.reshape(samples, latent_dim)

	cat_codes = np.asarray([digit for _ in range(samples)])
	cat_codes = to_categorical(cat_codes, num_classes=cat)

	z_input = np.hstack((z_latent, cat_codes))
	return [z_input, cat_codes]


def load_real_samples():
	(trainX, _), (_, _) = mnist.load_data()

	x = np.expand_dims(trainX, axis=-1)
	x = x.astype("float32")
	# scale from [0,255] to [-1,1]
	x = (x - 127.5) / 127.5
	return x

# select real samples
def generate_real_samples(dataset, samples):
	idx = np.random.randint(0, dataset.shape[0], samples)

	x = dataset[idx]
	y = np.ones((samples, 1))
	return x, y

# generate fake samples
def generate_fake_samples(generator, latent_dim, cat, samples):
	z_input, _ = generate_latent_points(latent_dim, cat, samples)

	images = generator.predict(z_input)
	y = np.zeros((samples, 1))
	return images, y

def summarize_performance(step, g_model, gan_model, latent_dim, cat, samples=100):
	x, _ = generate_fake_samples(g_model, latent_dim, cat, samples)

	# scale from [-1,1] to [0,1]
	x = (x+1)/2.0

	for i in range(100):
		plt.subplot(10, 10, i+1)
		plt.axis("off")
		plt.imshow(x[i, :, :, 0], cmap="gray_r")

	fname = "summary/generated_plot_{:04d}.png".format(step+1)
	plt.savefig(fname)
	plt.close()

	# save generator model
	fname2 = "models/model_{:04d}.h5".format(step+1)
	g_model.save(fname2)

	# save gan model
	fname3 = "models/gan_model_{:04d}.h5".format(step+1)
	gan_model.save(fname3)
	print("[INFO] Saved {}, {}, {}".format(fname, fname2, fname3))

def create_plot(examples, samples):
	for i in range(samples):
		plt.subplot(math.sqrt(samples), math.sqrt(samples), i+1)
		plt.axis("off")
		plt.imshow(examples[i, :, :, 0], cmap="gray_r")
	plt.savefig("created_plot.png")
	plt.close()