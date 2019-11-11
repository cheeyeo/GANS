from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, UpSampling2D, Conv2D, Reshape

model = Sequential()

# Create 128 5*5 activations with 100 dim random input vector...
model.add(Dense(128 * 5 * 5, input_dim=100))

# Resgape activations into 128 feature maps with size 5*5
model.add(Reshape((5, 5, 128)))

# Double to 128 10*10 feature maps
model.add(UpSampling2D())


# Output single image of 10*10 with padding=same
model.add(Conv2D(1, (3,3), padding="same"))

model.summary()