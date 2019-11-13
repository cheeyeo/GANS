## DCGAN for MNIST

Example GAN model from chapter 10 of "Generative Adversarial Network" book

## Extensions

* Update the example to use the Tanh activation function in the generator and scale all pixel values in the range [-1, 1]

* Update the example to use a larger or smaller latent space and compare the quality of the results and speed of training

* Update the discriminator and/or generator to make use of batch normalization

* Update the example to use one-sided label smoothing when training the discriminator, specifically change the target label of real examples from 1.0 to 0.9, and review the effects on image quality and speed of training

* Update the model configuration to use deeper or more shallow discriminator and/or generator models, perhaps experiment with the UpSampling2D layer in generator