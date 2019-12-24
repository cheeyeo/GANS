# Load generator model and generate images
import numpy as np
from keras.models import load_model
from utils import generate_latent_points
from utils import generate_latent_points_with_digit
from utils import create_plot
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, help="Path to model file")
ap.add_argument("-s", "--samples", type=int, help="Number of samples to generate")
ap.add_argument("-d", "--digit", type=int, help="Category control code to generate specific digit.")
args = vars(ap.parse_args())
print(args)

model = load_model(args["model"])
model.summary()

cat = 10
# Use same latent dim as during training
latent_dim = 62
# Number of samples to plot
samples = args["samples"]

if args["digit"] is None:
	z_input, _ = generate_latent_points(latent_dim, cat, samples)
else:
	z_input, _ = generate_latent_points_with_digit(latent_dim, cat, samples, args["digit"])

X = model.predict(z_input)
# scale from [-1,1] to [0,1]
X = (X+1) / 2.0

create_plot(X, samples)