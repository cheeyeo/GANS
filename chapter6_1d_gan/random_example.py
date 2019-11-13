# Generating random samples from X^2

import numpy as np
from matplotlib import pyplot as plt


def generate_samples(n=100):
	# generate random inputs in range [-0.5, 0.5]
	X1 = np.random.rand(n) - 0.5

	# generate outputs X^2
	X2 = X1*X1

	X1 = X1.reshape(n, 1)
	X2 = X2.reshape(n, 1)

	return np.hstack((X1, X2))


data = generate_samples()
plt.scatter(data[:, 0], data[:, 1])
plt.savefig("plot2.png")