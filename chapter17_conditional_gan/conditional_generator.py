
import numpy as np
from utils2 import generate_latent_points
from utils2 import save_plot
from keras.models import load_model

model = load_model("models/cgan_generator.h5")

latent_points, labels = generate_latent_points(100, 100)
print(labels.shape)
labels = np.asarray([x for _ in range(10) for x in range(10)])
print(labels)

X = model.predict([latent_points, labels])
# scale from [-1,1] to [0,1]
X = (X+1)/2.0

save_plot(X, 10)