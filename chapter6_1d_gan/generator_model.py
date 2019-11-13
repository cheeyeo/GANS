from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

def generate_latent_points(latent_dim, n):
	x_input = np.random.randn(latent_dim * n)
	x_input = x_input.reshape((n, latent_dim))
	return x_input

def generate_fake_samples(generator, latent_dim, n):
	x_input = generate_latent_points(latent_dim, n)

	X = generator.predict(x_input)

	plt.scatter(X[:, 0], X[:, 1])
	plt.savefig("generator_test.png")



def define_generator(latent_dim, n_outputs=2):
	model = Sequential()
	model.add(Dense(15, activation="relu", kernel_initializer="he_uniform", input_dim=latent_dim))
	model.add(Dense(n_outputs, activation="linear"))
	return model

if __name__ == "__main__":
	# Specify latent space of 5 dim
	latent_dim = 5
	model = define_generator(latent_dim)
	# model.summary()

	# plot_model(model, to_file="generator_plot.png", show_shapes=True, show_layer_names=True)

	generate_fake_samples(model, latent_dim, 100)