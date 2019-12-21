from model import define_generator, define_discriminator, define_gan
from utils import load_real_samples, generate_real_samples, generate_fake_samples, generate_latent_points
import numpy as np

def train(generator, discriminator, gan, dataset, latent_dim, epochs=100, batch=128):
	batch_per_epoch = int(dataset.shape[0]/batch)

	half_batch = int(batch/2)

	for i in range(epochs):
		for j in range(batch_per_epoch):
			X_real, y_real = generate_real_samples(dataset, half_batch)

			d_loss1, _ = discriminator.train_on_batch(X_real, y_real)

			X_fake, y_fake = generate_fake_samples(generator, latent_dim, half_batch)

			d_loss2, _ = discriminator.train_on_batch(X_fake, y_fake)

			Xgan = generate_latent_points(latent_dim, batch)
			ygan = np.ones((batch, 1))
			g_loss = gan.train_on_batch(Xgan, ygan)

			print("{:d}, {:d}/{:d}, d1={:.3f}, d2={:.3f}, g={:.3f}".format(i+1, j+1, batch_per_epoch, d_loss1, d_loss2, g_loss))

	generator.save("models/generator.h5")

if __name__ == "__main__":
	latent_dim = 100

	generator = define_generator(latent_dim)
	discriminator = define_discriminator()

	gan_model = define_gan(generator, discriminator)

	dataset = load_real_samples()

	train(generator, discriminator, gan_model, dataset, latent_dim)