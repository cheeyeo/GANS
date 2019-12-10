## Develop a Least Square GAN

From chapter 15 of book "Generative Adversarial Network"

## Notes

* Extension to the GAN architecture that addresses the problem of vannishing gradients and loss saturation

* Provides a signal to the generator about fake samples that are far from the discriminator's decision boundary for classifying them as real / fake

	Points which are generated far from deicison boundary provide little gradient information to generator on how to generate better images

	The small gradient for generated images far from decision boundary referred to as "vanishing gradient or loss saturation"

	Hence using binary cross entropy in the discriminator alone is unable to give a strong signal on how to best update the generator model



the further the generated images are from decision boundary, the larger the error signal provided to the generator, hence encouraging it to generate more realistic images

* LSGAN implemented with a minor change to output layer of discriminator model with the use of L2 loss function

* LSGAN is an extension to the GAN architecture where we change the loss function for discriminator from binary crossentropy to least squares loss

	Motivation is that least squares loss will penalize generated images based on their distance from decision boundary

	Hence, provides strong signal for generated images that are very different or far from existing data and addresses the problem of saturated loss


## Issues

Creating and running the examples still result in convergence failure i.e. the discriminator loss falls close to 0

The resulting images don't resemble any output from the training set ...