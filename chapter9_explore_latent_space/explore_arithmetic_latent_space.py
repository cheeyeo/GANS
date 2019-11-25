# Perform vector arithmetic in latent space

import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def generate_latent_points(latent_dim, n):
	X = np.random.randn(latent_dim * n)

	X = X.reshape((n, latent_dim))

	return X

def plot_generated(examples, rows, cols):
	for i in range(rows*cols):
		plt.subplot(rows, cols, i+1)
		plt.axis("off")
		plt.imshow(examples[i, :, :])
	plt.savefig("artifacts/generated_arthimetic_sample.png")

# Calculates average of latent space vectors
def average_points(points, ix):
	zero_ix = [i-1 for i in ix]

	vectors = points[zero_ix]

	avg_vector = np.mean(vectors, axis=0)

	all_vectors = np.vstack((vectors, avg_vector))

	return all_vectors


if __name__ == "__main__":
	model = load_model("artifacts/generator_model_020.h5")

	# latent_points = generate_latent_points(100, 100)

	# Save latent vectors
	# np.savez_compressed("latent_points.npz", latent_points)

	# X = model.predict(latent_points)

	# scale from [-1,1] to [0,1]
	# X =(X+1)/2.0

	# plot_generated(X, 10)

	# The indices are based on the image locations generated above; first image from top left is 1 etc
	smiling_woman_idx = [92, 93, 95]
	neutral_woman_idx = [96, 87, 90]
	neutral_man_idx = [10, 20, 32]
	data = np.load("latent_points.npz")
	points = data["arr_0"]

	# average vectors
	smiling_woman = average_points(points, smiling_woman_idx)
	neutral_woman = average_points(points, neutral_woman_idx)
	neutral_man = average_points(points, neutral_man_idx)

	all_vectors = np.vstack((smiling_woman, neutral_woman, neutral_man))

	images = model.predict(all_vectors)
	images = (images+1)/2.0
	plot_generated(images, 3, 4)

	# smiling_woman - neutral_woman + neutral_man = smiling_man
	result_vector = smiling_woman[-1] - neutral_woman[-1] + neutral_man[-1]

	result_vector = np.expand_dims(result_vector, 0)

	result_img = model.predict(result_vector)
	result_img = (result_img+1)/2.0
	plt.figure()
	plt.axis("off")
	plt.imshow(result_img[0])
	plt.savefig("artifacts/generated_arthimetic_result.png")
