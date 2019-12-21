# Same utility functions as in utils.py but for the conditional gans

import numpy as np
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

def load_real_samples():
	(trainX, trainY), (_, _) = fashion_mnist.load_data()
	X = np.expand_dims(trainX, axis=-1)
	X = X.astype("float32")
	# scale from [0,255] to [-1,1]
	X = (X-127.5) / 127.5
	return [X, trainY]

def generate_real_samples(dataset, sample):
	images, labels = dataset

	idx = np.random.randint(0, images.shape[0], sample)
	X, labels = images[idx], labels[idx]
	y = np.ones((sample, 1))

	return [X, labels], y


def generate_latent_points(latent_dim, sample, classes=10):
	x_input = np.random.randn(latent_dim * sample)
	z_input = x_input.reshape((sample, latent_dim))
	labels = np.random.randint(0, classes, sample)
	return [z_input, labels]

def generate_fake_samples(generator, latent_dim, sample):
  z_input, labels = generate_latent_points(latent_dim, sample)
  images = generator.predict([z_input, labels])
  y = np.zeros((sample, 1))
  return [images, labels], y

def save_plot(examples, n):
  for i in range(n*n):
    plt.subplot(n, n, i+1)
    plt.axis("off")
    plt.imshow(examples[i,:,:,0], cmap="gray_r")
  plt.savefig("conditional_generated_plot.png")