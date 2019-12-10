import numpy as np
from scipy.linalg import sqrtm

def calculate_fid(act1, act2):
	"""
	Calculte FID scores between 2 activations
	"""

	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

	# Sum squared difference of means..
	ssdiff = np.sum((mu1-mu2)**2.0)

	# sqrt of product between covariance matrix
	covmean = sqrtm(sigma1.dot(sigma2))

	if np.iscomplexobj(covmean):
		covmean = covmean.real

	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

	return fid

if __name__ == "__main__":
	# Test with 2 random collections of activations
	act1 = np.random.random(10*2048)
	act1 = act1.reshape((10, 2048))

	act2 = np.random.random(10*2048)
	act2 = act2.reshape((10, 2048))

	# calculate fid between act1 and act1
	# should return 0
	fid = calculate_fid(act1, act1)
	print("FID (same) :{:.3f}".format(fid))

	# FID between act1 and act2
	# Should return large number...
	fid = calculate_fid(act1, act2)
	print("FID (different) : {:.3f}".format(fid))