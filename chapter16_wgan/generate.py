from keras.models import load_model
from data import generate_latent_points
from data import plot_generated

model = load_model("models/model_1940.h5")
latent_points = generate_latent_points(100, 25)
X = model.predict(latent_points)
plot_generated(X, 5)