# Training script for GAN model

from model import define_discriminator, define_generator, define_gan, define_discriminator2, define_generator2
import numpy as np
import matplotlib.pyplot as plt

def load_real_samples(fname="img_align_celeba.npz"):
	data = np.load(fname)
	X = data["arr_0"]
	X = X.astype("float32")

	# scale from [0,255] to [-1, 1]
	X = (X - 127.5) / 127.5

	return X

def generate_real_samples(dataset, n):
	idx = np.random.randint(0, dataset.shape[0], n)

	X = dataset[idx]

	y = np.ones((n, 1))

	return X, y

def generate_latent_points(latent_dim, n):
	X = np.random.randn(latent_dim * n)

	X = X.reshape((n, latent_dim))

	return X

def generate_fake_samples(g_model, latent_dim, n):
	x = generate_latent_points(latent_dim, n)

	X = g_model.predict(x)

	y = np.zeros((n, 1))

	return X, y

def save_plot(examples, epoch, n=10):
	# scale from [-1, 1] to [0, 1]
	examples = (examples + 1) / 2.0

	for i in range(n*n):
		plt.subplot(n, n, i+1)
		plt.axis("off")
		plt.imshow(examples[i])

	filename = "artifacts/generated_plot_e{:03d}.png".format(epoch+1)
	plt.savefig(filename)
	plt.close()

def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n=100):
	
	X_real, y_real = generate_real_samples(dataset, n)

	_, acc_real = d_model.evaluate(X_real, y_real)

	X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n)

	_, acc_fake = d_model.evaluate(X_fake, y_fake)

	print("[INFO] Real accuracy: {:.0f}%, Fake accuracy: {:.0f}%".format(acc_real*100, acc_fake*100))

	save_plot(X_fake, epoch)

	filename = "artifacts/generator_model_{:03d}.h5".format(epoch+1)
	g_model.save(filename)

def train(g_model, d_model, gan_model, dataset, latent_dim, epochs=100, batch=128):

	# Batch per epoch => 50000 / 128
	batch_per_epoch = int(dataset.shape[0] / batch)

	half_batch = int(batch / 2)

	for i in range(epochs):
		for j in range(batch_per_epoch):
			X_real, y_real = generate_real_samples(dataset, half_batch)

			d_loss1, _ = d_model.train_on_batch(X_real, y_real)

			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

			d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)

			X_gan = generate_latent_points(latent_dim, batch)

			y_gan = np.ones((batch, 1))

			g_loss = gan_model.train_on_batch(X_gan, y_gan)

			print("[INFO] Epoch: {:d}, {:d}/{:d}, Real loss={:.3f}, Fake loss={:.3f}, Gan loss={:3f}".format(i+1, j+1, batch_per_epoch, d_loss1, d_loss2, g_loss))

		if (i+1) % 10 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)


if __name__ == "__main__":
	latent_dim = 100
	epochs = 100
	batch_size = 128

	d_model = define_discriminator2()

	g_model = define_generator2(latent_dim)

	gan_model = define_gan(g_model, d_model)

	dataset = load_real_samples()
	print(dataset.min(), dataset.max())

	train(g_model, d_model, gan_model, dataset, latent_dim, epochs=epochs, batch=batch_size)