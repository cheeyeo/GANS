import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(trainX, trainY), (testX, testY) = mnist.load_data()
print("trainX shape: {}, trainY shape: {}".format(trainX.shape, trainY.shape))
print("testX shape: {}, testY shape: {}".format(testX.shape, testY.shape))

for i in range(25):
	plt.subplot(5, 5, i+1)
	plt.axis("off")
	plt.imshow(trainX[i], cmap="gray_r")
plt.savefig("artifacts/mnist_examples.png")