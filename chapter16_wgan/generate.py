from tensorflow.keras.models import load_model
from data import generate_latent_points
from data import plot_generated

model = load_model("models/model_0970.h5")
latent_points = generate_latent_points(50, 25)
X = model.predict(latent_points)
plot_generated(X, 5)