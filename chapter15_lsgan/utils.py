from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

def load_real_samples():
	(trainX, _), (_, _) = mnist.load_data()

	X = np.expand_dims(trainX, axis=-1)

	X = X.astype("float32")

	# Scale from [0-255] to [-1, 1]
	X = (X - 127.5) / 127.5

	return X

def generate_real_samples(dataset, samples):
	idx = np.random.randint(0, dataset.shape[0], samples)

	X = dataset[idx]
	y = np.ones((samples, 1))
	return X, y


def generate_latent_points(latent_dim, samples):
	x_input = np.random.randn(latent_dim * samples)

	x_input = x_input.reshape(samples, latent_dim)

	return x_input

def generate_fake_samples(generator, latent_dim, samples):
	x = generate_latent_points(latent_dim, samples)

	x = generator.predict(x)
	y = np.zeros((samples, 1))

	return x, y

def summarize_performance(step, g_model, latent_dim, samples=100):
	X, _ = generate_fake_samples(g_model, latent_dim, samples)

	# Scale from [-1, 1] to [0,1]
	X = (X+1)/2.0

	for i in range(10*10):
		plt.subplot(10, 10, i+1)
		plt.axis("off")
		plt.imshow(X[i, :, :, 0], cmap="gray_r")

	fname = "artifacts/generated_plot_{:06d}.png".format(step+1)
	plt.savefig(fname)
	plt.close()

	mname = "models/model_{:06d}.h5".format(step+1)
	g_model.save(mname)
	print("Saved {}, {}".format(fname, mname))

def plot_history(d1_hist, d2_hist, g_hist):
	plt.plot(d1_hist, label="dloss1")
	plt.plot(d2_hist, label="dloss2")
	plt.plot(g_hist, label="gloss")
	plt.legend()
	fname = "artifacts/plot_line_plot_loss.png"
	plt.savefig(fname)
	plt.close()
	print("Saved {}".format(fname))

def plot_generated(examples, n):
	for i in range(n*n):
		plt.subplot(n, n, i+1)
		plt.axis("off")
		plt.imshow(examples[i, :, :, 0], cmap="gray_r")
	plt.savefig("generator_plot.png")
	plt.close()