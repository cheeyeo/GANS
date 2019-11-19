# Define the combined generator and discriminator model
from discriminator_model import discriminator_model
from generator_model import generator_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

def gan_model(generator, discriminator):
	discriminator.trainable = False

	model = Sequential()
	model.add(generator)
	model.add(discriminator)

	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss="binary_crossentropy", optimizer=opt)
	return model


if __name__ == "__main__":
	latent_dim = 100

	d_model = discriminator_model()

	g_model = generator_model(latent_dim)

	gan_model = gan_model(g_model, d_model)

	gan_model.summary()

	plot_model(gan_model, to_file="artifacts/gan_model.png", show_shapes=True, show_layer_names=True)