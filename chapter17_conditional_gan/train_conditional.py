# Training conditional GAN
import numpy as np
from keras.utils import plot_model
from utils2 import load_real_samples, generate_real_samples, generate_fake_samples, generate_latent_points
from model import conditional_generator, conditional_discriminator, conditional_gan

def train(g_model, d_model, gan_model, dataset, latent_dim, epochs=100, batch=128):
	batch_per_epoch = int(dataset[0].shape[0] / batch)

	half_batch = int(batch/2)

	for i in range(epochs):
		for j in range(batch_per_epoch):
			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)

			d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)

			[X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

			d_loss2, _ = d_model.train_on_batch([X_fake, labels_fake], y_fake)

			[z_input, labels] = generate_latent_points(latent_dim, batch)

			y_gan = np.ones((batch, 1))

			g_loss = gan_model.train_on_batch([z_input, labels], y_gan)

			print("{:d}, {:d}/{:d}, d1={:.3f}, d2={:.3f}, g={:.3f}".format(i+1, j+1, batch_per_epoch, d_loss1, d_loss2, g_loss))

	g_model.save("models/cgan_generator.h5")

if __name__ == "__main__":
	latent_dim = 100

	d_model = conditional_discriminator()
	plot_model(d_model, show_shapes=True, to_file="conditional_discriminator.png")

	g_model = conditional_generator(latent_dim)
	plot_model(g_model, show_shapes=True, to_file="conditional_generator.png")

	gan_model = conditional_gan(g_model, d_model)
	plot_model(gan_model, show_shapes=True, to_file="conditional_gan.png")

	dataset = load_real_samples()

	train(g_model, d_model, gan_model, dataset, latent_dim)