# Train GAN model
from discriminator_model import discriminator_model
from discriminator_model import generate_real_samples
from discriminator_model import load_real_samples
from generator_model import generator_model
from generator_model import generate_fake_samples
from generator_model import generate_latent_points
from gan_model import gan_model
import numpy as np
import matplotlib.pyplot as plt

def save_plot(examples, epoch, n=7):
	# scale from [-1, 1] to [0,1]
	examples = (examples + 1) / 2.0

	for i in range(n*n):
		plt.subplot(n, n, i+1)
		plt.axis("off")
		plt.imshow(examples[i])
	filename = "artifacts/generated_plot_e{:03d}.png".format(epoch+1)
	plt.savefig(filename)
	plt.close()


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, samples=150):
	X_real, y_real = generate_real_samples(dataset, samples)

	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)

	X_fake, y_fake = generate_fake_samples(g_model, latent_dim, samples)

	_, acc_fake = d_model.evaluate(X_fake, y_fake, verbose=0)

	print("Accuacy real: {:0f}, fake: {:0f}".format(acc_real * 100, acc_fake * 100))

	save_plot(X_fake, epoch)

	fname = "models/generator_model_{:03d}.h5".format(epoch+1)
	g_model.save(fname)

def train(g_model, d_model, gan_model, dataset, latent_dim, epochs=200, batch=128):
	
	batch_per_epoch = int(dataset.shape[0] / batch)

	half_batch = int(batch/2)

	for i in range(epochs):
		for j in range(batch_per_epoch):
			X_real, y_real = generate_real_samples(dataset, half_batch)

			d_loss1, _ = d_model.train_on_batch(X_real, y_real)

			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

			d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)

			X_gan = generate_latent_points(latent_dim, batch)

			# create inverted labels for fake samples..
			y_gan = np.ones((batch, 1))

			g_loss = gan_model.train_on_batch(X_gan, y_gan)

			print("> Epoch {:d}, {:d}/{:d}, d1={:.3f}, d2={:.3f}, g={:.3f}".format(i+1, j+1, batch_per_epoch, d_loss1, d_loss2, g_loss))

		if (i+1)%10 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)

if __name__ == "__main__":

	latent_dim = 100

	d_model = discriminator_model()

	g_model = generator_model(latent_dim)

	gan_model = gan_model(g_model, d_model)

	dataset = load_real_samples()

	train(g_model, d_model, gan_model, dataset, latent_dim)