from tensorflow.keras.constraints import Constraint
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import RMSprop

# Clip model weights to given hypercube
class ClipConstraint(Constraint):
	def __init__(self, clip_value):
		self.clip_value = clip_value

	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)

	def get_config(self):
		return {"clip_value": self.clip_value}

def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)

def define_critic(input_shape=(28, 28, 1)):
	init = RandomNormal(stddev=0.02)

	# weight constraint
	constraint = ClipConstraint(0.01)

	model = Sequential()
	# downsample to 14x14
	model.add(Conv2D(64, (4,4), strides=(2,2), padding="same", kernel_initializer=init, kernel_constraint=constraint, input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	# downsample to 7x7
	model.add(Conv2D(64, (4,4), strides=(2,2), padding="same", kernel_initializer=init, kernel_constraint=constraint))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	model.add(Flatten())
	model.add(Dense(1))

	opt = RMSprop(lr=0.00005)
	model.compile(loss=wasserstein_loss, optimizer=opt)
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

	# output 28x28x1
	model.add(Conv2D(1, (7,7), activation="tanh", padding="same", kernel_initializer=init))

	return model

def define_gan(generator, critic):
	critic.trainable = False

	model = Sequential()
	model.add(generator)
	model.add(critic)

	opt = RMSprop(lr=0.00005)
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model

