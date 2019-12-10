## Inception Score

Chapter 12 of book "Generative Adversarial Network"

## Notes

* Inception Score is an objective metric for evaluating quality of generated images, specifically images output by GAN

* Inception Score involves using pre-trained DL neural network for image classification to classify generated images; Inception V3 model is used as classifier

* Generated images are classified using the Inception V3 model; the probs of each image belonging to a class is predicted; predictions summarized into inception score

* Intent of inception score is to capture 2 properties of a collection of generated images:

	* Image Quality 

	* Image Diversity

* Inception score has lowest value of 1.0 and highest value of number of classes supported by classification model e.g. for ILSVRC 2012 dataset with 1000 classes, the highest inception score on the dataset will be 1000

* Issue with inception score as evaluation metric:

  Appropriate for generated images of objects known to the model used to calculate the conditional class probs i.e. generated image from GAN generator must belong to one of these classes of objects used to train the Inception model...

  Also need to ensure generated images are square and need to be rescaled to 300x300 

  There also needs to be a good distribution of generated images across all classes and an even number of examples for each class

