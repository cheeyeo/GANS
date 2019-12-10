# Define the discriminator, generator and LSGAN models

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam

def define_discriminator(input_shape=(28, 28, 1)):
	init = RandomNormal(stddev=0.02)

	model = Sequential()

	# downsample to 14x14
	model.add(Conv2D(64, (4, 4), strides=(2,2), padding="same", kernel_initializer=init, input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	# downsample to 7x7
	model.add(Conv2D(128, (4,4), strides=(2,2), padding="same", kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	# classifier
	model.add(Flatten())
	model.add(Dense(1, activation="linear", kernel_initializer=init))

	model.compile(loss="mse", optimizer=Adam(lr=0.0002, beta_1=0.5))

	return model

def define_generator(latent_dim):
	init = RandomNormal(stddev=0.02)

	model = Sequential()

	nodes = 256 * 7 * 7

	model.add(Dense(nodes, kernel_initializer=init, input_dim=latent_dim))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(Reshape((7, 7, 256)))

	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same", kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(Activation("relu"))

	# upsample to 28x28
	model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding="same", kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(Activation("relu"))

	model.add(Conv2D(1, (7,7), padding="same", kernel_initializer=init))
	model.add(Activation("tanh"))

	return model


# Create composite model for training
def define_gan(generator, discriminator):
	discriminator.trainable = False

	model = Sequential()
	model.add(generator)
	model.add(discriminator)

	model.compile(loss="mse", optimizer=Adam(lr=0.0002, beta_1=0.5))
	return model
