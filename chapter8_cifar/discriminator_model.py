from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import numpy as np
from tensorflow.keras.datasets import cifar10


def load_real_samples():
	(trainX, _), (_, _) = cifar10.load_data()

	X = trainX.astype("float32")

	# Scale from [0,255] to [-1,1]
	X = (X - 127.5)/127.5
	return X

def generate_real_samples(dataset, samples):
	idx = np.random.randint(0, dataset.shape[0], samples)

	X = dataset[idx]
	y = np.ones((samples, 1))

	return X, y

def generate_fake_samples(samples):
	X = np.random.randn(32 * 32 * 3 * samples)
	X = -1 + X * 2
	X = X.reshape((samples, 32, 32, 3))
	y = np.zeros((samples, 1))
	return X, y

def train_discriminator(model, dataset, iter=20, batch=128):
	half_batch = int(batch/2)

	for i in range(iter):
		X_real, y_real = generate_real_samples(dataset, half_batch)
 
		_, real_acc = model.train_on_batch(X_real, y_real)

		# generate fake examples
		X_fake, y_fake = generate_fake_samples(half_batch)

		_, fake_acc = model.train_on_batch(X_fake, y_fake)

		print("{:d} epoch, Real={:.0f}%, Fake={:.0f}%".format(i+1, real_acc*100, fake_acc*100))

def discriminator_model(in_shape=(32, 32, 3)):
	model = Sequential()

	# Normal
	model.add(Conv2D(64, (3,3), padding="same", input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))

	# Downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	# Downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	# DOwnsample
	model.add(Conv2D(256, (3,3), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation="sigmoid"))

	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss="binary_crossentropy", 
								optimizer=opt,
								metrics=["accuracy"])

	return model

if __name__ == "__main__":
	m = discriminator_model()
	m.summary()

	plot_model(m, to_file="artifacts/discriminator_model.png", show_shapes=True, show_layer_names=True)

	dataset = load_real_samples()
	train_discriminator(m, dataset)