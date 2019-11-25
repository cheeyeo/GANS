from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

# Discriminator model
def define_discriminator(input_shape=(80, 80, 3)):
	model = Sequential()

	# Normal
	model.add(Conv2D(128, (5,5), padding="same", input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	# Downsample to 40x40
	model.add(Conv2D(128, (5,5), strides=(2,2), padding="same"))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	# Downsample to 20x20
	model.add(Conv2D(128, (5,5), strides=(2,2), padding="same"))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	# Downsample to 10x10
	model.add(Conv2D(128, (5,5), strides=(2,2), padding="same"))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	# Downsample to 5x5
	model.add(Conv2D(128, (5,5), strides=(2,2), padding="same"))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation="sigmoid"))

	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

	return model

def define_discriminator2(input_shape=(80, 80, 3)):
	model = Sequential()

	# Normal
	model.add(Conv2D(64, (3,3), padding="same", input_shape=input_shape))
	model.add(LeakyReLU(alpha=0.2))

	# Downsample to 40x40
	model.add(Conv2D(128, (3,3), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	# Downsample to 20x20
	model.add(Conv2D(128, (3,3), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	# Downsample to 10x10
	model.add(Conv2D(128, (3,3), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	# Downsample to 5x5
	model.add(Conv2D(128, (3,3), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation="sigmoid"))

	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

	return model


# Create generator model
def define_generator(latent_dim):
	init = RandomNormal(mean=0.0, stddev=0.2)
	model = Sequential()

	n_nodes = 128 * 5 * 5

	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((5, 5, 128)))

	# UPsample to 10x10
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same", kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	# Upsample to 20x20
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same", kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	# Upsample to 40x40
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same", kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	# Upsample to 80x80
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same", kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	# Output layer 80x80x3
	model.add(Conv2D(3, (5,5), activation="tanh", padding="same", kernel_initializer=init))

	return model

def define_generator2(latent_dim):
	init = RandomNormal(mean=0.0, stddev=0.2)
	model = Sequential()

	n_nodes = 256 * 5 * 5

	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((5, 5, 256)))

	# UPsample to 10x10
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	# Upsample to 20x20
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	# Upsample to 40x40
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	# Upsample to 80x80
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	# Output layer 80x80x3
	model.add(Conv2D(3, (3,3), activation="tanh", padding="same", kernel_initializer=init))

	return model



# Overall GAN model
def define_gan(g_model, d_model):
	# Discriminator trained separately; mark weights as not trainable to ensure that only weights of generator model updated;

	# Only affects when training combined GAN model not when training discriminator alone
	d_model.trainable = False

	model = Sequential()
	model.add(g_model)
	model.add(d_model)

	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss="binary_crossentropy", optimizer=opt)
	return model


if __name__ == "__main__":
	d_model = define_discriminator2()
	d_model.summary()
	print()

	g_model = define_generator2(100)
	g_model.summary()

	gan = define_gan(g_model, d_model)
	# gan.summary()