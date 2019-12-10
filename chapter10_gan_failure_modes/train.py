# Train stable GAN model

import numpy as np
from model import *
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

def load_real_samples():
	(trainX, trainY), (_, _) = mnist.load_data()

	X = np.expand_dims(trainX, axis=-1)

	selected_idx = trainY == 8

	X = X[selected_idx]
	X = X.astype("float32")
	# scale from [0,255] tp [-1, 1]
	X = (X - 127.5) / 127.5

	return X

def generate_real_samples(dataset, n):

	idx = np.random.randint(0, dataset.shape[0], n)

	X = dataset[idx]

	y = np.ones((n, 1))

	return X, y

def generate_latent_points(latent_dim, n):

	x_input = np.random.randn(latent_dim * n)

	x_input = x_input.reshape(n, latent_dim)

	return x_input

def generate_fake_samples(generator, latent_dim, n):
	x_input = generate_latent_points(latent_dim, n)

	X = generator.predict(x_input)
	y = np.zeros((n, 1))

	return X, y

# Generate samples and run model evaluation

def summarize_performance(step, g_model, latent_dim, samples=100):
	X, _ = generate_fake_samples(g_model, latent_dim, samples)

	# scale from [-1, 1] to [0,1]
	X = (X+1)/2.0

	for i in range(10*10):
		plt.subplot(10, 10, i+1)
		plt.axis("off")
		plt.imshow(X[i, :, :, 0], cmap="gray_r")

	plt.savefig("results_collapse/generated_plot_{:03d}.png".format(step+1))
	plt.close()
	g_model.save("results_collapse/model_{:03d}.h5".format(step+1))

def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):

	plt.subplot(2, 1, 1)
	plt.plot(d1_hist, label="d-real")
	plt.plot(d2_hist, label="d-fake")
	plt.plot(g_hist, label="gan")
	plt.legend()

	# plot discriminator accuracy
	plt.subplot(2, 1, 2)
	plt.plot(a1_hist, label="acc-real")
	plt.plot(a2_hist, label="acc-fake")
	plt.legend()

	# save plot to file
	plt.savefig("results_collapse/plot_line_plot_loss.png")
	plt.close()

def train(g_model, d_model, gan_model, dataset, latent_dim, epochs=10, batch=128):

	print(dataset.shape)

	batch_per_epoch = int(dataset.shape[0] / batch)

	steps = batch_per_epoch * epochs

	half_batch = int(batch/2)

	d1_hist = list()
	d2_hist = list()
	g_hist = list()
	a1_hist = list()
	a2_hist = list()

	for i in range(steps):
		X_real, y_real = generate_real_samples(dataset, half_batch)

		d_loss1, d_acc1 = d_model.train_on_batch(X_real, y_real)

		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

		d_loss2, d_acc2 = d_model.train_on_batch(X_fake, y_fake)

		X_gan = generate_latent_points(latent_dim, batch)

		y_gan = np.ones((batch, 1))

		g_loss = gan_model.train_on_batch(X_gan, y_gan)

		print("[INFO] Epoch: {:d}, d1={:.3f}, d2={:.3f}, g={:.3f}, a1={:d}, a2={:d}".format(i+1, d_loss1, d_loss2, g_loss, int(100*d_acc1), int(100*d_acc2)))

		d1_hist.append(d_loss1)
		d2_hist.append(d_loss2)
		g_hist.append(g_loss)
		a1_hist.append(d_acc1)
		a2_hist.append(d_acc2)

		if (i+1) % batch_per_epoch==0:
			summarize_performance(i, g_model, latent_dim)

	plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)


if __name__ == "__main__":
	latent_dim = 1

	d = define_discriminator()

	g = define_generator(latent_dim)

	gan_model = define_gan(g, d)

	dataset = load_real_samples()
	print(dataset.shape)
	print(dataset.min(), dataset.max())

	train(g, d, gan_model, dataset, latent_dim)