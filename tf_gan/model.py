# Defines the discriminator and generator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.losses import BinaryCrossentropy

def make_generator(latent_dim=100):
	model = Sequential()

	nodes = 256 * 7 * 7
	model.add(Dense(nodes, use_bias=False, input_shape=(latent_dim,)))
	model.add(BatchNormalization())
	model.add(LeakyReLU())
	
	model.add(Reshape((7, 7, 256)))
	assert model.output_shape == (None, 7, 7, 256)

	model.add(Conv2DTranspose(128, (4,4), strides=(1,1), padding="same", use_bias=False))
	assert model.output_shape == (None, 7, 7, 128)
	model.add(BatchNormalization())
	model.add(LeakyReLU())

	# Upsample to 14x14x64
	model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding="same", use_bias=False))
	assert model.output_shape == (None, 14, 14, 64)
	model.add(BatchNormalization())
	model.add(LeakyReLU())

	# Upsample to 28x28x1
	model.add(Conv2DTranspose(1, (4,4), strides=(2,2), padding="same", use_bias=False, activation="tanh"))
	assert model.output_shape == (None, 28, 28, 1)

	return model

def make_discriminator(input_shape=(28, 28, 1)):
	model = Sequential()

	# Downsample to 14x14x64
	model.add(Conv2D(64, (4,4), strides=(2,2), padding="same", input_shape=input_shape))
	assert model.output_shape == (None, 14, 14, 64)
	model.add(LeakyReLU())
	model.add(Dropout(0.3))

	# DOwnsample to 7x7x128
	model.add(Conv2D(128, (4,4), strides=(2,2), padding="same", input_shape=input_shape))
	assert model.output_shape == (None, 7, 7, 128)
	model.add(LeakyReLU())
	model.add(Dropout(0.3))

	model.add(Flatten())
	model.add(Dense(1))
	return model

cross_entropy = BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
	"""
	Compares discriminator prediction for real images to 1 and fake images to 0
	"""

	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	total_loss = real_loss + fake_loss
	return total_loss

def generator_loss(fake_output):
	"""
	Compare discriminator's prediction on fake images to 1
	"""
	return cross_entropy(tf.ones_like(fake_output), fake_output)



if __name__ == "__main__":
	g = make_generator(latent_dim=100)
	g.summary()

	d = make_discriminator()
	d.summary()