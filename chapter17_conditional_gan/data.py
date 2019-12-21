# Load and visualize Fashion MNIST dataset
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
print("X shape: {}, y shape: {}".format(trainX.shape, trainY.shape))
print("Test X shape: {}, y shape: {}".format(testX.shape, testY.shape))

for i in range(100):
	plt.subplot(10, 10, i+1)
	plt.axis("off")
	plt.imshow(trainX[i], cmap="gray_r")
plt.savefig("data.png")