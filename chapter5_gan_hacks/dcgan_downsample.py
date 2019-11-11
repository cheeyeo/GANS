# Downsample using strided convolutions

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

model = Sequential()

# half the size of input to 32x32 due to use of strides = (2,2)
model.add(Conv2D(64, (3,3), strides=(2,2), padding="same", input_shape=(64, 64, 3)))

model.summary()
