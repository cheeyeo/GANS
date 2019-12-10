from model import define_discriminator
from model import define_generator
from model import define_gan
from utils import load_real_samples
from utils import generate_real_samples
from utils import generate_fake_samples
from utils import generate_latent_points
from utils import summarize_performance
from utils import plot_history
import numpy as np

def train(g_model, d_model, gan_model, dataset, latent_dim, epochs=20, batch=64):
	batch_per_epoch = int(dataset.shape[0] / batch)

	steps = batch_per_epoch * epochs

	half_batch = int(batch / 2)

	d1_hist, d2_hist, g_hist = list(), list(), list()

	for i in range(steps):
		X_real, y_real = generate_real_samples(dataset, half_batch)

		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

		# update discriminator model
		d_loss1 = d_model.train_on_batch(X_real, y_real)
		d_loss2 = d_model.train_on_batch(X_fake, y_fake)

		# update generator via discriminator's loss
		z_input = generate_latent_points(latent_dim, batch)
		y_real2 = np.ones((batch, 1))

		g_loss = gan_model.train_on_batch(z_input, y_real2)

		print("{:d}, d1={:.3f}, d2={:.3f}, g={:.3f}".format(i+1, d_loss1, d_loss2, g_loss))

		d1_hist.append(d_loss1)
		d2_hist.append(d_loss2)
		g_hist.append(g_loss)

		if (i+1) % (batch_per_epoch * 1) == 0:
			summarize_performance(i, g_model, latent_dim)

	plot_history(d1_hist, d2_hist, g_hist)


if __name__ == "__main__":
	latent_dim = 100

	discriminator = define_discriminator()

	generator = define_generator(latent_dim)

	gan_model = define_gan(generator, discriminator)

	dataset = load_real_samples()
	print(dataset.shape)

	train(generator, discriminator, gan_model, dataset, latent_dim)