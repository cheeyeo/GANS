from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU

model = Sequential()

model.add(Conv2D(64, (3,3), strides=(2,2), padding="same", input_shape=(64, 64, 3)))

model.add(BatchNormalization())

model.add(LeakyReLU(0.2))

model.summary()