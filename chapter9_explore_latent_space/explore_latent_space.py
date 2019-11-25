# Explore the latent space by loading pre-built model
# and using linear interpolation

import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Spherical Linear Interpolation Function
def slerp(val, low, high):
	"""
	The latent space is a high dimensional hypersphere / multimodal Gaussian distribution and the curvature of sphere needs to be accounted for when performing linear interpolation between 2 points
	"""

	omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))

	so = np.sin(omega)

	if so == 0:
		# LERP?
		return (1.0-val) * low + val * high

	return np.sin((1.0-val)*omega)/so * low + np.sin(val*omega)/so * high



# Create interpolation path between 2 points and generate faces along the path
def interpolate_points(p1, p2, steps=10):
	"""
	Using linear interpolation to calculate ratios of contributions from 2 given points, and return series of interpolated vectors for each ratio
	"""

	ratios = np.linspace(0, 1, num=steps)

	vectors = list()
	for ratio in ratios:
		v = slerp(ratio, p1, p2)
		vectors.append(v)

	return np.asarray(vectors)


def generate_latent_points(latent_dim, n):
	X = np.random.randn(latent_dim * n)

	X = X.reshape((n, latent_dim))

	return X

def plot_generated(examples, n):
	for i in range(n*n):
		plt.subplot(n, n, i+1)
		plt.axis("off")
		plt.imshow(examples[i, :, :])
	plt.savefig("artifacts/generated_linear_interpolation.png")


if __name__ == "__main__":
	model = load_model("artifacts/generator_model_020.h5")

	n = 20

	pts = generate_latent_points(100, n)

	results = None

	for i in range(0, n, 2):
		interpolated = interpolate_points(pts[i], pts[i+1])

		X = model.predict(interpolated)

		# scale from [-1,1] to [0,1]
		X = (X+1)/2.0

		if results is None:
			results = X
		else:
			results = np.vstack((results, X))

	plot_generated(results, 10)