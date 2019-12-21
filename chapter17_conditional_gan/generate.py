from utils import generate_latent_points
from utils import show_plot
from keras.models import load_model

model = load_model("models/generator.h5")

latent_points = generate_latent_points(100, 100)

X = model.predict(latent_points)

show_plot(X, 10)