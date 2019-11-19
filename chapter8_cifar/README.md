## GAN for CIFAR-10 dataset

Tutorial from chapter 8 of book "Generative Adversarial Network"

## Extensions

* Change the size of latent space to be larger or smaller and compare the quality of results and speed of training

* Update discriminator/generator model to make use of BatchNormalization

* Update example to use one-sided label smoothing when training the discriminator, changing the target label of real examples from 1.0 to 0.9 and add random noise, then review effects on image quality and speed of training

* Update model configuration to use deeper or more shallow discriminator/generator models, perhaps experiment with UpSampling2D layers in generator