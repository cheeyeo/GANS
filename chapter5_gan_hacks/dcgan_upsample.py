# Scale input to desired output by stacking transpose convolutional layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2DTranspose


model = Sequential()

# Below uses upsample strided convolutions by setting strides to (2,2); uupsamples output from 64x64 to 128x128
model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding="same", input_shape=(64, 64, 3)))

model.summary()