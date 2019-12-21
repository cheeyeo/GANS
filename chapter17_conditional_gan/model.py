from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.optimizers import Adam
from keras.utils import plot_model

# Unconditioned DCGAN
def define_discriminator(input_shape=(28, 28, 1)):
	model = Sequential()

	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding="same", input_shape=input_shape))
	model.add(LeakyReLU(alpha=0.2))

	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	# FC layer
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation="sigmoid"))

	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

	return model

def define_generator(latent_dim):
	model = Sequential()

	# Set up 7x7 image
	nodes = 7*7*128
	model.add(Dense(nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))

	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(1, (7,7), padding="same"))
	model.add(Activation("tanh"))

	return model

def define_gan(generator, discriminator):
	discriminator.trainable = False

	model = Sequential()
	model.add(generator)
	model.add(discriminator)

	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss="binary_crossentropy", optimizer=opt)
	return model

def conditional_discriminator(in_shape=(28, 28, 1), classes=10):
	in_label = Input(shape=(1,))

	# embedding for categorical input
	li = Embedding(classes, 50)(in_label)

	nodes = in_shape[0] * in_shape[1]
	li = Dense(nodes)(li)
	li = Reshape((in_shape[0], in_shape[1], 1))(li)

	in_image = Input(shape=in_shape)

	merge = Concatenate()([in_image, li])

	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding="same")(merge)
	fe = LeakyReLU(alpha=0.2)(fe)

	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding="same")(fe)
	fe = LeakyReLU(alpha=0.2)(fe)

	fe = Flatten()(fe)

	fe = Dropout(0.4)(fe)

	out_layer = Dense(1, activation="sigmoid")(fe)

	model = Model(inputs=[in_image, in_label], outputs=out_layer)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

	return model

def conditional_generator(latent_dim, classes=10):
	in_label = Input(shape=(1,))

	li = Embedding(classes, 50)(in_label)

	nodes = 7*7
	li = Dense(nodes)(li)
	# reshape to additional channel
	li = Reshape((7, 7, 1))(li)

	# image generator input
	in_lat = Input(shape=(latent_dim,))

	nodes = 7*7*128
	gen = Dense(nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((7,7,128))(gen)

	merge = Concatenate()([gen, li])

	# upsample
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding="same")(merge)
	gen = LeakyReLU(alpha=0.2)(gen)

	# upsample
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding="same")(gen)
	gen = LeakyReLU(alpha=0.2)(gen)

	# FC layer
	out_layer = Conv2D(1, (7,7), padding="same")(gen)
	out_layer = Activation("tanh")(out_layer)

	model = Model(inputs=[in_lat, in_label], outputs=out_layer)
	return model

def conditional_gan(generator, discriminator):
	discriminator.trainable = False

	# get noise and label inputs from generator
	gen_noise, gen_label = generator.input

	# image output from generator
	gen_output = generator.output

	# connect generator output and class label as input to discriminator
	gan_output = discriminator([gen_output, gen_label])

	model = Model(inputs=[gen_noise, gen_label], outputs=gan_output)

	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss="binary_crossentropy", optimizer=opt)

	return model



if __name__ == "__main__":
	m = conditional_discriminator()
	m.summary()
	plot_model(m, show_shapes=True, to_file="discriminator.png")