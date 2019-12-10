from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

def load_real_samples():
	(trainX, trainY), (_, _) = mnist.load_data()

	selected_idx = trainY==7

	X = trainX[selected_idx]
	X = np.expand_dims(X, axis=-1)
	X = X.astype("float32")

	# scale from [0,255] to [-1,1]
	X = (X-127.5)/127.5
	return X

def generate_real_samples(dataset, samples):
	idx = np.random.randint(0, dataset.shape[0], samples)
	X = dataset[idx]
	# -1 for real
	y = -np.ones((samples, 1))
	return X, y

def generate_latent_points(latent_dim, samples):
	x_input = np.random.randn(latent_dim * samples)
	x_input = x_input.reshape(samples, latent_dim)
	return x_input

def generate_fake_samples(generator, latent_dim, samples):
	x_input = generate_latent_points(latent_dim, samples)
	X = generator.predict(x_input)
	# 1 for fake
	y = np.ones((samples, 1))
	return X, y

def summarize_performance(step, g_model, latent_dim, samples=100):

	X, _ = generate_fake_samples(g_model, latent_dim, samples)

	# scale from [-1, 1] to [0,1]
	X = (X+1)/2.0

	for i in range(10*10):
		plt.subplot(10, 10, i+1)
		plt.axis("off")
		plt.imshow(X[i,:,:,0], cmap="gray_r")

	fname = "artifacts/generated_plot_{:04d}.png".format(step+1)
	plt.savefig(fname)
	mname = "models/model_{:04d}.h5".format(step+1)
	g_model.save(mname)
	print("Saved: {}, {}".format(fname, mname))

def plot_history(d1_hist, d2_hist, g_hist):
	plt.figure()
	plt.plot(d1_hist, label="crit_real")
	plt.plot(d2_hist, label="crit_fake")
	plt.plot(g_hist, label="gen")
	plt.legend()
	plt.savefig("artifacts/loss.png")
	plt.close()

def plot_generated(examples, n):
	for i in range(n*n):
		plt.subplot(n, n, i+1)
		plt.axis("off")
		plt.imshow(examples[i, :, :, 0], cmap="gray_r")
	plt.savefig("generated.png")