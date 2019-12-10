## Frechet Inception Distance

Chapter 13 of book "Generative Adversarial Network"


## Notes

* Calculate distance between feature vectors for real and generated images

* Goal of FID is to evaluate synthetic images based on stats of a collection of synthetic images compared to stats of a collection of real images from target domain

* FID summarizes the distance between Inception features for real and generated images in same domain


* Inception score only measures the quality of a collection of synthetic images based on how well Inception V3 classifies them; it combines both confidence of class predictions (quality) and integral of marginal probability of predicted classes (diversity) but not how well it compares to real images

* FID score uses the inception v3 model ut uses the last pooling layer as a feature extractor to capture features of an input image;

these activations calculated for collection of real and generated images

activations summarized as multivariate gaussian by calculating mean and covariance of images

distance between these 2 distributions (real, generated) calculated using Frechet distance ( Wasserstein-2 distance)

* Lower FID indicates better quality images; higher FID score indicates lower quality image
