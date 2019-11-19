from tensorflow.keras.datasets import cifar10
from matplotlib import pyplot as plt

(trainX, trainY), (testX, testY) = cifar10.load_data()
print("X shape: {}, y shape: {}".format(trainX.shape, trainY.shape))
print("X test shape: {}, y test shape: {}".format(testX.shape, testY.shape))

for i in range(49):
	plt.subplot(7, 7, i+1)
	plt.axis("off")
	plt.imshow(trainX[i])
plt.savefig("artifacts/cifar10_sample.png")