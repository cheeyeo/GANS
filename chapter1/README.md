* Generative modelling using deep learning methods such as CNN

* Generative modelling is an unsupervised learning task in ML that involves automatically discovering and learning patterns in input data that the model can be used to generate new examples from original dataset

GANs trains a generative model by framing the problem as a supervised learning problem with 2 submodels:

the generator model trained to generate new examples and the discriminator model that tries to classify examples as real or fake

two models trained together in adversarial zero-sum game until discriminator fooled about half the time, meaninig generator creating plausible examples

GANs are an architecture

Generative models => approaches that explictly or implictly model the distribution of inputs as well as outputs as sampling from them can generate synthetic data in input space

Examples of generative models:
* Naive Bayes
* LDA
* GMM ( Gaussian Mixture Model)
* RBM ( Restricted Boltzmann Machinne )
* Deep Belief Network (DBN)
* Variational Autoencoder (VAE)
* Generative Adversarial Network (GAN)

GAN => deep learning based generative model

model architecture for training generative model


Most GANs today based on DCGAN architecture


Latent variables / latent space => projection or compression of a data distribution


In GANs, generator model applies meaning to points in chosen latent space, such that new points drawn from latent space can be provided as input to generator model and used to generate new output examples

RANDOM INPUT VECTOR
        ||
  Generator Model
        ||
  Generated Example

Discriminator model takes an input and predicts a binary class label of real / fake

* Supervised learning => data samples have an input X and an output label y

* Unsupervised learning => no output label y; algorithm tries to learn patterns inherent in data

* Discriminative modelling: 
  i.e. classification

  develop model to predict class label given examples of input variables

  known as discriminative modelling as model must discriminate examples of inputs across classes; must make a decision on what class an example belongs to


 * Generative Modelling:

   Unsupervised models that summarize distribution of input variables

   may be used to create / generate new examples in input distribution

   					| MODEL |
   					   ||
   					| Generated Example |

   eg. a variable may belong to a Gaussian distribution; generative model may be able to sufficiently summarize this data distribution and then generate new examples to fit into the distribution of input variable







