from discriminator_model import define_discriminator_model, generate_real_samples, load_real_samples
from generator_model import define_generator, generate_fake_samples, generate_latent_points
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

def define_gan_model(g_model, d_model):
	d_model.trainable = False

	model = Sequential()
	model.add(g_model)
	model.add(d_model)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
	return model

def save_plot(examples, epoch, n=10):
	for i in range(n*n):
		plt.subplot(n, n, i+1)
		plt.axis("off")
		plt.imshow(examples[i, :, :, 0], cmap="gray_r")
	filename = "artifacts/generated_plot_e{:03d}.png".format(epoch+1)
	plt.savefig(filename)
	plt.close()


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, samples=100):

	X_real, y_real = generate_real_samples(dataset, samples)

	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)

	X_fake, y_fake = generate_fake_samples(g_model, latent_dim, samples)

	_, acc_fake = d_model.evaluate(X_fake, y_fake, verbose=0)

	print("[INFO] Real Acc: {:.0f}%, Fake Acc: {:.0f}%".format(acc_real * 100, acc_fake*100))

	save_plot(X_fake, epoch)

	filename = "models/generator_model_{:03d}.h5".format(epoch+1)
	g_model.save(filename)


def train(g_model, d_model, gan_model, dataset, latent_dim, epochs=100, batch=256):
	
	batch_per_epoch = int(dataset.shape[0]/batch)

	half = int(batch/2)

	for i in range(epochs):
		for j in range(batch_per_epoch):
			X_real, y_real = generate_real_samples(dataset, half)

			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half)

			X = np.vstack((X_real, X_fake))

			y = np.vstack((y_real, y_fake))

			# update discriminator model weights
			d_loss, _ = d_model.train_on_batch(X, y)

			X_gan = generate_latent_points(latent_dim, batch)

			y_gan = np.ones((batch, 1))

			g_loss, _ = gan_model.train_on_batch(X_gan, y_gan)

			print("[INFO] {:d}, {:d}/{:d}, d={:.3f}, g={:.3f}".format(i+1, j+1, batch_per_epoch, d_loss, g_loss))

		if (i+1) % 10 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)


if __name__ == "__main__":
	latent_dim = 100

	d_model = define_discriminator_model()

	g_model = define_generator(latent_dim)

	gan_model = define_gan_model(g_model, d_model)

	gan_model.summary()

	plot_model(gan_model, to_file="artifacts/gan_model.png", show_shapes=True, show_layer_names=True)

	dataset = load_real_samples()

	train(g_model, d_model, gan_model, dataset, latent_dim)