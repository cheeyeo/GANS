# Example model of a stable GAN

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam

def define_discriminator(input_shape=(28, 28, 1)):
	init = RandomNormal(stddev=0.02)
	model = Sequential()

	# Downsample to 14x14
	model.add(Conv2D(64, (4,4), strides=(2,2), padding="same", kernel_initializer=init, input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	# DOwnsample to 7x7
	model.add(Conv2D(64, (4,4), strides=(2,2), padding="same", kernel_initializer=init, input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	model.add(Flatten())
	model.add(Dense(1, activation="sigmoid"))
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss="binary_crossentropy",
								optimizer=opt,
								metrics=["accuracy"])

	return model

def define_generator(latent_dim):
	init = RandomNormal(stddev=0.02)

	model = Sequential()

	nodes = 128 * 7 * 7
	model.add(Dense(nodes, kernel_initializer=init, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))

	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same", kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same", kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	# output of 28x28x1
	model.add(Conv2D(1, (7,7), activation="tanh", padding="same", kernel_initializer=init))

	return model

def define_gan(g_model, d_model):
	d_model.trainable = False

	model = Sequential()
	model.add(g_model)
	model.add(d_model)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss="binary_crossentropy", optimizer=opt)
	return model