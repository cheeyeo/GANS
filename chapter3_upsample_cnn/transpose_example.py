import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2DTranspose

X = np.array([[1,2], [3,4]])
print(X)

# reshape input into sample, width, height, channel
X = X.reshape((1, 2, 2, 1))

model = Sequential()
model.add(Conv2DTranspose(1, (1,1), strides=(2,2), input_shape=(2, 2, 1)))
model.summary()

# Define dummy weights to print put output
weights = [np.asarray([[[[1]]]]), np.asarray([0])]

model.set_weights(weights)

yhat = model.predict(X)

yhat = yhat.reshape((4, 4))

print(yhat)

# [[1. 0. 2. 0.]
#  [0. 0. 0. 0.]
#  [3. 0. 4. 0.]
#  [0. 0. 0. 0.]]
