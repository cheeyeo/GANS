from tensorflow.keras.models import load_model
from utils import generate_latent_points
from utils import plot_generated

model = load_model("models/model_018740.h5")

latent_points = generate_latent_points(100, 100)

X = model.predict(latent_points)

plot_generated(X, 10)