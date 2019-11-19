# Use generator model to create an image

from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

def generate_latent_points(latent_dim, samples):
	x_input = np.random.randn(latent_dim * samples)

	x_input = x_input.reshape(samples, latent_dim)

	return x_input

def save_plot(examples, n):
	for i in range(n*n):
		plt.subplot(n, n, i+1)
		plt.axis("off")
		plt.imshow(examples[i, :, :])
	plt.show()

if __name__ == "__main__":
	model = load_model("models/generator_model_200.h5")

	latent_points = generate_latent_points(100, 100)

	X = model.predict(latent_points)

	# scale from [-1, 1] to [0,1]
	X = (X+1) / 2.0

	save_plot(X, 10)