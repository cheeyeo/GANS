## Information Maximizing GAN / InfoGAN

From chapter 18 of "Generative Adversarial Network" book

* Extension to GAN architecture that introduces control variables that are automatically learnt by the architecture and allow control over generated image such as style, thickness in the case of MNIST dataset

* InfoGAN is motivated by desire to disentangle and control the properties in generated images.

* InfoGAN involves the addition of control variables to generate an auxiliary model that predicts the control variables, trained via mutual information loss

* Main aim is to disentangle and control properties in generated images

* InfoGAN is an information-theoretic extension to GAN that is able to learn disentangled representations in an unsupervised manner.

* InfoGAN involves the addition of control variables to generate an auxiliary model that predicts the control variables; auxiliary model trained through mutual information loss

* Provide control variables as input to generator along with point in latent space ( noise ); generator trained to use control variables to influence specific properties of generated images

* InfoGAN motivated by desire to disentangle properties of generated images e.g. in the case of faces, properties of a face can be disentangled and controlled such as shape of face, hairstyle etc

* Control variables are provided with noise as input to generator and model trained using mutual information loss

Maximize mutual information between small subset of GAN's noise variables and observations

Mutual information refers to amount of information learnt about one variable given another variable

Interested in information about control variables given the generated image using noise and control variables

Mutual Information between X and Y I(X;Y) measures amount of information learnt from knowledge of random variable Y about other random variable X


MI is calculated as conditional entropy of image ( created by generator G from noise Z and control variable c), given the control variables (c) subtracted from marginal entropy of control variables (c); e.g.

MI = Entropy(c) - Entropy(c | G(z,c))

* Training the generator via mutual information is achieved through use of a new model, known as Q or auxiliary model

* The auxiliary model shares the same weights as discriminator but the auxiliary model predicts the control codes used to generate the image

* Both discriminator and auxiliary models are used to update generator; first to improve the likelihood of generating images that fool the discriminator; secondly, to improve mutual information between control codes used to generate an image and the auxiliary model's prediction of the control codes

* Result is that the generator model is regularized via mutual information loss such that the control codes capture salient properties of generated images and in turn, used to control image generation process

* Mutual information can be used when we are interested in learning a parameterized mapping from given input X to a higher level representation Y which preserves information about original input; task of maximizing mutual information is equivalent to training an autoencoder to minimize reconstruction error

## Implement mutual information loss function

2 main types of control variables used with InfoGAN:

* Categorical

* Continuous

In Keras, easier to simplify control variables to categorical (one-hot encode) and continuous variables to Gaussian/Uniform distribution and have them as separate outputs on auxiliary model for each control variable type

Above so that different loss function can be used for each, simplifying implementation

### Categorical Control Variables

Used to control type / class of generated image

One-hot encoded vector
i.e. 10 class values and control code is one class 6, then the representation is [0,0,0,0,0,1,0,0,0,0]

Don't choose categorical control variables when training model; generated randomly e.g. each selected with uniform probability for each sample

uniform categorical distribution on latent codes c ~ Cat(K=10, p=0.1)

output layer of auxiliary model for categorical variable one-hot encoded vector to match input control code and softmax activation used

Entropy of control variable is a constant close to 0; conditional entropy can be calculated directly as cross-entropy loss between control variable input and output from auxiliary model

hence use categorical cross-entropy loss function

### Continuous Control Variable

Control style of image

sampled from uniform distribution between -1 and 1

input to generator model

auxiliary model implement the prediction of continuous control variables with Gaussian distribution where output layer configured to have one node for mean, one node for std dev of Gaussian i.e. 2 outputs for each continuous variable

nodes that output mean can use linear activation function

nodes that output std dev can use sigmoid between 0 and 1

