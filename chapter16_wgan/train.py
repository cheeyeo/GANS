import numpy as np
from model import define_critic, define_generator, define_gan
from data import generate_real_samples, generate_fake_samples, generate_latent_points, summarize_performance, plot_history, load_real_samples
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", default=10, type=int, help="Number of epochs to train for.")
ap.add_argument("-l", "--latent_dim", default=50, type=int, help="Latent dim for generator")
ap.add_argument("-b", "--batch_size", default=64, type=int, help="Sets the batch size.")
ap.add_argument("-c", "--critic", default=5, type=int, help="Number of cycles to train critic for. Default to 5 as set in paper.")

args = vars(ap.parse_args())

def train(g_model, c_model, gan_model, dataset, latent_dim, epochs=10, batch=64, n_critic=5):

	batch_per_epoch = int(dataset.shape[0]/batch)
	print("[INFO] BATCH PER EPOCH: {:d}".format(batch_per_epoch))

	steps = batch_per_epoch * epochs
	print("[INFO] TOTAL STEPS: {:d}".format(steps))

	half_batch = int(batch/2)

	c1_hist, c2_hist, g_hist = list(), list(), list()

	for i in range(steps):
		c1_tmp, c2_tmp = list(), list()

		for _ in range(n_critic):
			X_real, y_real = generate_real_samples(dataset, half_batch)

			c_loss1 = c_model.train_on_batch(X_real, y_real)
			c1_tmp.append(c_loss1)

			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

			c_loss2 = c_model.train_on_batch(X_fake, y_fake)
			c2_tmp.append(c_loss2)

		c1_hist.append(np.mean(c1_tmp))
		c2_hist.append(np.mean(c2_tmp))

		X_gan = generate_latent_points(latent_dim, batch)
		y_gan = -np.ones((batch, 1))
		g_loss = gan_model.train_on_batch(X_gan, y_gan)
		g_hist.append(g_loss)
		print("{:d}, c1={:.3f}, c2={:.3f}, g={:.3f}".format(i+1, c1_hist[-1], c2_hist[-1], g_loss))

		if (i+1) % batch_per_epoch == 0:
			summarize_performance(i, g_model, latent_dim)

	plot_history(c1_hist, c2_hist, g_hist)

if __name__ == "__main__":
	print(args)
	latent_dim = args["latent_dim"]
	batch_size = args["batch_size"]
	n_critic = args["critic"]
	epochs = args["epochs"]

	critic = define_critic()

	generator = define_generator(latent_dim)

	gan_model = define_gan(generator, critic)

	dataset = load_real_samples()
	print(dataset.shape)

	train(generator, critic, gan_model, dataset, latent_dim, epochs=epochs, batch=batch_size, n_critic=n_critic)