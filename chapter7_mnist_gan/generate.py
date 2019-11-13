# Use the pretrained GAN model to generate new mnist images...

from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


def generate_latent_points(latent_dim, samples):
	x = np.random.randn(latent_dim * samples)

	x = x.reshape(samples, latent_dim)

	return x


def save_plot(examples, n):
	for i in range(n*n):
		plt.subplot(n, n, i+1)
		plt.axis("off")
		plt.imshow(examples[i, :, :, 0], cmap="gray_r")
	plt.savefig("artifacts/generated_examples.png")

if __name__ == "__main__":
	model = load_model("models/generator_model_100.h5")

	latent_points = generate_latent_points(100, 25)

	X = model.predict(latent_points)

	save_plot(X, 5)

	# Generate a single example using a single vector of values...
	vector = np.asarray([[0.0 for _ in range(100)]])

	X2 = model.predict(vector)
	plt.figure()
	plt.imshow(X2[0, :, :, 0], cmap="gray_r")
	plt.savefig("artifacts/generated_single_example.png")