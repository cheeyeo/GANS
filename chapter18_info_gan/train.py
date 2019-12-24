import numpy as np
from utils import summarize_performance, generate_real_samples, generate_fake_samples, generate_latent_points, load_real_samples
from model import define_generator, define_discriminator, define_gan

def train(g_model, d_model, gan_model, dataset, latent_dim, cat, epochs=100, batch=64):
	batch_per_epoch = int(dataset.shape[0] / batch)

	steps = batch_per_epoch * epochs

	half_batch = int(batch / 2)

	for i in range(steps):
		X_real, y_real = generate_real_samples(dataset, half_batch)

		d_loss1 = d_model.train_on_batch(X_real, y_real)

		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, cat, half_batch)

		d_loss2 = d_model.train_on_batch(X_fake, y_fake)

		z_input, cat_codes = generate_latent_points(latent_dim, cat, batch)
		y_gan = np.ones((batch, 1))

		_, g_1, g_2 = gan_model.train_on_batch(z_input, [y_gan, cat_codes])

		print("[INFO] {:d}, d[{:.3f}, {:.3f}], g[{:.3f}], q[{:.3f}]".format(i+1, d_loss1, d_loss2, g_1, g_2))

		if (i+1) % (batch_per_epoch * 10) == 0:
			summarize_performance(i, g_model, gan_model, latent_dim, cat)

if __name__ == "__main__":
	# Nos of categories
	cat = 10

	# Dimension of latent space
	latent_dim = 62

	d_model, q_model = define_discriminator(cat)

	gen_input_size = latent_dim + cat
	g_model = define_generator(gen_input_size)

	gan_model = define_gan(g_model, d_model, q_model)

	dataset = load_real_samples()

	train(g_model, d_model, gan_model, dataset, latent_dim, cat)

