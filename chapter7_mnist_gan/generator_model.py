from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Activation

def define_generator_tanh(latent_dim):
	n_nodes = 128*7*7
	model = Sequential()
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(Activation("tanh"))
	model.add(Reshape((7, 7, 128)))

	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"))
	model.add(Activation("tanh"))

	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"))
	model.add(Activation("tanh"))

	model.add(Conv2D(1, (7,7), activation="sigmoid", padding="same"))

	return model


def define_generator(latent_dim):
	n_nodes = 128 * 7 * 7
	model = Sequential()
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	# Upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(1, (7,7), activation="sigmoid", padding="same"))

	return model

def generate_latent_points(latent_dim, samples):
	x_input = np.random.randn(latent_dim*samples)
	x_input = x_input.reshape(samples, latent_dim)
	return x_input


def generate_fake_samples(model, latent_dim, samples):
	x_input = generate_latent_points(latent_dim, samples)
	X = model.predict(x_input)
	# Fake class labels
	y = np.zeros((samples, 1))
	return X, y


if __name__ == "__main__":
	# Set latent dim
	latent_dim = 100

	model = define_generator(latent_dim)

	model.summary()

	plot_model(model, to_file="artifacts/generator_model.png", show_shapes=True, show_layer_names=True)

	samples = 25

	X, _ = generate_fake_samples(model, latent_dim, samples)

	for i in range(samples):
		plt.subplot(5, 5, i+1)
		plt.axis("off")
		plt.imshow(X[i, :, :, 0], cmap="gray_r")

	plt.savefig("artifacts/generator_subplot.png")