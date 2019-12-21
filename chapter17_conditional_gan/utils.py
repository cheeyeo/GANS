import numpy as np
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

def load_real_samples():
	(trainX, _), (_, _) = fashion_mnist.load_data()
	X = np.expand_dims(trainX, axis=-1)
	X = X.astype("float32")
	# scale from [0,255] to [-1,1]
	X = (X-127.5) / 127.5
	return X

def generate_real_samples(dataset, sample):
	idx = np.random.randint(0, dataset.shape[0], sample)
	X = dataset[idx]
	y = np.ones((sample, 1))
	return X, y

def generate_latent_points(latent_dim, sample):
	x_input = np.random.randn(latent_dim * sample)
	x_input = x_input.reshape((sample, latent_dim))
	return x_input

def generate_fake_samples(generator, latent_dim, sample):
  x_input = generate_latent_points(latent_dim, sample)
  X = generator.predict(x_input)
  y = np.zeros((sample, 1))
  return X, y

def show_plot(examples, n):
  for i in range(n*n):
    plt.subplot(n, n, i+1)
    plt.axis("off")
    plt.imshow(examples[i,:,:,0], cmap="gray_r")
  plt.savefig("generated_plot.png")