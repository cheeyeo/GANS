# Example of label smoothing where the labels vary slightly more or less than 1.0 for real and slightly more than 0.0 for fake images

import numpy as np

# Smooth class=1 to [0.7, 1.2]
def smooth_positive_labels(y):
	return y - 0.3 + (np.random.random(y.shape) * 0.5)

# Smooth class=0 to [0.0, 0.3]
def smooth_negative_labels(y):
	return y + np.random.random(y.shape) * 0.3


if __name__ == "__main__":
	# generate real class labels
	n_samples = 1000
	y = np.ones((n_samples, 1))
	y = smooth_positive_labels(y)
	print("Smooth positive labels...")
	print(y.shape, y.min(), y.max())
	print()

	y = np.zeros((n_samples, 1))
	y = smooth_negative_labels(y)
	print("Smooth negative labels...")
	print(y.shape, y.min(), y.max())