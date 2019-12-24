# InfoGAN models
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.initializers import RandomNormal
from keras.utils import plot_model


def define_generator(gen_input_size):
	init = RandomNormal(stddev=0.02)

	in_lat = Input(shape=(gen_input_size,))

	nodes = 512 * 7 * 7
	gen = Dense(nodes, kernel_initializer=init)(in_lat)
	gen = Activation("relu")(gen)
	gen = BatchNormalization()(gen)
	gen = Reshape((7, 7, 512))(gen)

	# normal FC
	gen = Conv2D(128, (4,4), padding="same", kernel_initializer=init)(gen)
	gen = Activation("relu")(gen)
	gen = BatchNormalization()(gen)

	# upsample to 14x14
	gen = Conv2DTranspose(64, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(gen)
	gen = Activation("relu")(gen)
	gen = BatchNormalization()(gen)

	# upsample to 28x28
	gen = Conv2DTranspose(1, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(gen)

	out_layer = Activation("tanh")(gen)

	model = Model(inputs=in_lat, outputs=out_layer)
	return model

def define_discriminator(cat, in_shape=(28, 28, 1)):
	init = RandomNormal(stddev=0.02)

	# image input
	in_image = Input(shape=in_shape)

	# downsample to 14x14
	d = Conv2D(64, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.1)(d)

	# downsample to 7x7
	d = Conv2D(128, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(d)
	d = LeakyReLU(alpha=0.1)(d)
	d = BatchNormalization()(d)

	# Normal FC
	d = Conv2D(256, (4,4), padding="same", kernel_initializer=init)(d)
	d = LeakyReLU(alpha=0.1)(d)
	d = BatchNormalization()(d)

	d = Flatten()(d)

	out_classifier = Dense(1, activation="sigmoid")(d)

	d_model = Model(inputs=in_image, outputs=out_classifier)

	d_model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))

	# Create auxiliary / q model layers
	q = Dense(128)(d)
	q = BatchNormalization()(q)
	q = LeakyReLU(alpha=0.1)(q)

	out_codes = Dense(cat, activation="softmax")(q)
	# define q model
	q_model = Model(inputs=in_image, outputs=out_codes)

	return d_model, q_model

def define_gan(g_model, d_model, q_model):
	d_model.trainable = False

	# connect g outputs to d inputs
	d_output = d_model(g_model.output)

	# connect g output to q input
	q_output = q_model(g_model.output)

	model = Model(inputs=g_model.input, outputs=[d_output, q_output])

	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=["binary_crossentropy", "categorical_crossentropy"], optimizer=opt)

	return model


if __name__ == "__main__":
	cat = 10
	latent_dim = 62

	d_model, q_model = define_discriminator(cat)

	gen_input_size = latent_dim + cat
	g_model = define_generator(gen_input_size)
	plot_model(g_model, to_file="generator_plot.png", show_shapes=True, show_layer_names=True)

	gan_model = define_gan(g_model, d_model, q_model)
	gan_model.summary()

	plot_model(gan_model, to_file="gan_plot.png", show_shapes=True, show_layer_names=True)
