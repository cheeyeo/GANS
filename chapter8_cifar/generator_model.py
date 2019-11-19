# Define generator model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

# Generate points in latent space as input for generator
def generate_latent_points(latent_dim, samples):
	x = np.random.randn(latent_dim * samples)
	x = x.reshape(samples, latent_dim)
	return x

# Generate fake samples, with class labels
def generate_fake_samples(g_model, latent_dim, samples):
	x = generate_latent_points(latent_dim, samples)

	x = g_model.predict(x)
	y = np.zeros((samples, 1))

	return x, y

def generator_model(latent_dim):
	model = Sequential()

	n_nodes = 256 * 4 * 4
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((4, 4, 256)))

	# upsample to 8x8
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	# upsample to 16x16
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	# upsample to 32x32
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	# output layer
	model.add(Conv2D(3, (3,3), activation="tanh", padding="same"))

	return model

if __name__ == "__main__":
	latent_dim = 100

	m = generator_model(latent_dim)

	m.summary()

	plot_model(m, to_file="artifacts/generator_plot.png", show_shapes=True, show_layer_names=True)

	samples = 49
	X, _ = generate_fake_samples(m, latent_dim, samples)

	X = (X+1) / 2.0

	for i in range(samples):
		plt.subplot(7, 7, i+1)
		plt.axis("off")
		plt.imshow(X[i])
	plt.savefig("artifacts/generator_examples.png")