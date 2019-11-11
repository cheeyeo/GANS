import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import UpSampling2D


X = np.array([
	  [1,2],
	  [3,4]
	])

print(X)

# reshape X to be 1 sample, 2 rows, 2 cols, 1 channel to input into model..
X = X.reshape((1, 2, 2, 1))

model = Sequential()
model.add(UpSampling2D(input_shape=(2, 2, 1)))
model.summary()
# the UpSampling2D layer does not perform any learning hence has no weights or parameters, which will show as 0 in the summary...

yhat = model.predict(X)
yhat = yhat.reshape((4,4))
print(yhat)

# [[1. 1. 2. 2.]
# [1. 1. 2. 2.]
# [3. 3. 4. 4.]
# [3. 3. 4. 4.]]