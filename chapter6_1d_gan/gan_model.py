from tensorflow.keras.models import Sequential
from generator_model import define_generator
from discriminator_model import define_discriminator_model
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

def generate_real_samples(n):
	X1 = np.random.randn(n) - 0.5
	X2 = X1 * X1
	X1 = X1.reshape(n, 1)
	X2 = X2.reshape(n, 1)
	X = np.hstack((X1, X2))
	y = np.ones((n, 1))
	return X, y

def generate_latent_points(latent_dim, n):
	x_input = np.random.randn(latent_dim * n)
	x_input = x_input.reshape((n, latent_dim))
	return x_input

def generate_fake_samples(generator, latent_dim, n):
	x_input = generate_latent_points(latent_dim, n)
	X = generator.predict(x_input)
	y = np.zeros((n, 1))
	return X, y

def define_gan(generator, discriminator):
	# make weights in discriminator not trainable
	discriminator.trainable = False
	model = Sequential()
	model.add(generator)
	model.add(discriminator)
	model.compile(loss="binary_crossentropy", optimizer="adam")
	return model

def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
	# prepare real samples
	x_real, y_real = generate_real_samples(n)

	_, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)

	# generate fake samples
	x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
	_, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)

	print("[INFO] Epoch: {}, Real Acc: {}, Fake Acc: {}".format(epoch, acc_real, acc_fake))

	plt.scatter(x_real[:, 0], x_real[:, 1], color="red")
	plt.scatter(x_fake[:, 0], x_fake[:, 1], color="blue")
	fname = "generated_plot_e{:3d}.png".format(epoch+1)
	plt.savefig(fname)
	plt.close()



def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=128, n_eval=2000):

	half = int(n_batch / 2)

	for i in range(n_epochs):
		x_real, y_real = generate_real_samples(half)

		x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half)

		# update discriminator
		d_model.train_on_batch(x_real, y_real)
		d_model.train_on_batch(x_fake, y_fake)

		# prepare points in latent space as input for generator
		x_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for fake samples
		y_gan = np.ones((n_batch, 1))

		# update generator via discriminator's errors
		gan_model.train_on_batch(x_gan, y_gan)

		if (i+1) % n_eval == 0:
			summarize_performance(i, g_model, d_model, latent_dim)


latent_dim = 5

discriminator = define_discriminator_model()

generator = define_generator(latent_dim)

gan_model = define_gan(generator, discriminator)

gan_model.summary()

# plot_model(gan_model, to_file="gan_model.png", show_shapes=True, show_layer_names=True)

train(generator, discriminator, gan_model, latent_dim)