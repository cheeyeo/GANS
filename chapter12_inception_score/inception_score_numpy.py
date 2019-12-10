import numpy as np

def calculate_inception_score(p_yx, eps=1E-16):
	# calculate p(y)
	p_y = np.expand_dims(p_yx.mean(axis=0), 0)

	# KL divergence for each image
	kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))

	# sum over classes
	sum_kl_d = kl_d.sum(axis=1)

	# average over images
	avg_kl_d = np.mean(sum_kl_d)

	# undo logs?
	is_score = np.exp(avg_kl_d)

	return is_score


if __name__ == "__main__":
	# testing with sample probs with 3 classes of image and perfect confident prediction for each class of image...

	p_yx = np.asarray([[1.0, 0.0, 0.0],
										 [0.0, 1.0, 0.0],
										 [0.0, 0.0, 1.0]
										])

	score = calculate_inception_score(p_yx)
	print(score)