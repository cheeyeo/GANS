from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets.mnist import load_data
import numpy as np

def define_discriminator_model(in_shape=(28, 28, 1)):
	model = Sequential()
	# Use large strides to downsample input
	model.add(Conv2D(64, (3,3), strides=(2,2), padding="same", input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation="sigmoid"))

	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
	return model

def load_real_samples():
	(trainX, _), (_, _) = load_data()
	X = np.expand_dims(trainX, axis=-1)
	X = X.astype("float32")
	# scale from 0-255 to 0-1
	X = X / 255.0
	return X

def generate_real_samples(dataset, n_samples):
	idx = np.random.randint(0, dataset.shape[0], n_samples)
	X = dataset[idx]
	# Real class labels of 1
	y = np.ones((n_samples, 1))
	return X, y

def generate_fake_samples(n_samples):
	# generate uniform random nos in [0,1]
	X = np.random.randn(28 * 28 * n_samples)

	X = X.reshape((n_samples, 28, 28, 1))

	# Fake class labels of 0
	y = np.zeros((n_samples, 1))

	return X, y

def train_discriminator(model, dataset, iter=100, batch=256):
	half = int(batch/2)

	for i in range(iter):
		X_real, y_real = generate_real_samples(dataset, half)

		_, real_acc = model.train_on_batch(X_real, y_real)

		X_fake, y_fake = generate_fake_samples(half)

		_, fake_acc = model.train_on_batch(X_fake, y_fake)

		print("[INFO] Epoch: {:d} Real acc={:.0f}%, Fake acc={:.0f}%".format(i, real_acc*100, fake_acc*100))

if __name__ == "__main__":
	model = define_discriminator_model()
	model.summary()

	dataset = load_real_samples()

	plot_model(model, to_file="artifacts/discriminator_model.png", show_shapes=True, show_layer_names=True)
	
	train_discriminator(model, dataset)