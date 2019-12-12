from tensorflow.keras.datasets import mnist
import numpy as np

def load_real_data():
	(X_train, _), (_, _) = mnist.load_data()
	X_train = np.expand_dims(X_train, axis=-1)
	X_train = X_train.astype("float32")
	
	# Scale [0,255] to [-1, 1]
	X_train = (X_train - 127.5) / 127.5
	return X_train