import numpy as np

def scale_images(images):
	# Convert to float32 
	images = images.astype("float32")

	# scale from [0,255] to [-1, 1]
	images = (images - 127.5) / 127.5
	return images

images = np.random.randint(0, 256, 28*28*3)

images = images.reshape((1, 28, 28, 3))

print(images.min(), images.max())

scaled = scale_images(images)
print(scaled.min(), scaled.max())