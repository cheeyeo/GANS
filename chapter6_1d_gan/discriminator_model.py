from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
import numpy as np

def generate_real_samples(n):
	X1 = np.random.randn(n) - 0.5
	X2 = X1 * X1
	X1 = X1.reshape(n, 1)
	X2 = X2.reshape(n, 1)
	X = np.hstack((X1, X2))
	y = np.ones((n, 1))
	return X, y

def generate_fake_samples(n):
	X1 = -1 + np.random.randn(n) * 2
	X2 = -1 + np.random.randn(n) * 2
	X1 = X1.reshape(n, 1)
	X2 = X2.reshape(n, 1)
	X = np.hstack((X1, X2))
	y = np.zeros((n, 1))
	return X, y

def define_discriminator_model(inputs=2):
	model = Sequential()
	model.add(Dense(25, activation="relu", kernel_initializer="he_uniform", input_dim=inputs))
	model.add(Dense(1, activation="sigmoid"))
	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	return model

def train_discriminator_model(model, n_epochs=1000, n_batch=128):
	half_batch = int(n_batch / 2)

	for i in range(n_epochs):
		X_real, y_real = generate_real_samples(half_batch)

		model.train_on_batch(X_real, y_real)

		# generate fake samples
		X_fake, y_fake = generate_fake_samples(half_batch)

		model.train_on_batch(X_fake, y_fake)

		# evaluate model
		_, acc_real = model.evaluate(X_real, y_real, verbose=0)

		_, acc_fake = model.evaluate(X_fake, y_fake)
		print(i, acc_real, acc_fake)

if __name__ == "__main__":
	model = define_discriminator_model()
	# model.summary()
	# plot_model(model, to_file="discriminator_model.png", show_shapes=True, show_layer_names=True)
	train_discriminator_model(model)