import os
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from data import load_real_data
from model import make_generator, make_discriminator, generator_loss, discriminator_loss
import matplotlib.pyplot as plt

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('artifacts/image_at_epoch_{:04d}.png'.format(epoch))
  plt.close()


@tf.function
def train_step(images):
	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_images = generator(noise, training=True)

		real_output = discriminator(images, training=True)
		fake_output = discriminator(generated_images, training=True)

		gen_loss = generator_loss(fake_output)
		disc_loss = discriminator_loss(real_output, fake_output)

	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

	gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
	for epoch in range(epochs):
		start = time.time()

		for image_batch in dataset:
			train_step(image_batch)

		generate_and_save_images(generator, epoch+1, seed)

		if (epoch+1) % 15 == 0:
			checkpoint.save(file_prefix=checkpoint_prefix)

		print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

	generate_and_save_images(generator, epochs, seed)



BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Shuffle dataset
X_train = load_real_data()
training_data = tf.data.Dataset.from_tensor_slices(X_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

print(training_data)

# Define models, setups checkpoints..
generator = make_generator(latent_dim=100)
discriminator = make_discriminator()

generator_optimizer = Adam(1e-4)
discriminator_optimizer = Adam(1e-4)

checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.sep.join([checkpoint_dir, "ckpt"])
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)

train(training_data, EPOCHS)