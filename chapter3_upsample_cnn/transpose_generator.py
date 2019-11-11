from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose


model = Sequential()
model.add(Dense(128 * 5 * 5, input_dim=100))
model.add(Reshape((5, 5, 128)))
# Double input from 128 5*5 to 1 10x10 feature map
model.add(Conv2DTranspose(1, (3,3), strides=(2,2), padding="same"))

model.summary()