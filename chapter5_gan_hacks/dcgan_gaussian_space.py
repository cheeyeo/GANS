# Example on how to generate random samples from a latent space as input to generator model from a standard Gaussian distribution with 0 mean, std dev of 1


import numpy as np

def generate_latent_points(latent_dim, samples):
	x_input = np.random.randn(latent_dim * samples)

	x_input = x_input.reshape((samples, latent_dim))

	return x_input


n_dim = 100
samples = 500

samples = generate_latent_points(n_dim, samples)

print(samples.shape, samples.mean(), samples.std())