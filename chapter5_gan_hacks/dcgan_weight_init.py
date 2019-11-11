# Example of initializing model's weights using a Gaussian distribution

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.initializers import RandomNormal

model = Sequential()

init = RandomNormal(mean=0.0, stddev=0.02)

model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding="same", kernel_initializer=init, input_shape=(64, 64, 3)))

model.summary()