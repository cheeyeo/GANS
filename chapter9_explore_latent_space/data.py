## Preparing CelebA dataset
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN

def extract_face(model, pixels, required_size=(80, 80)):
  """
  Extract only the face from the image
  """

  faces = model.detect_faces(pixels)

  if len(faces) == 0:
    return None

  x1, y1, width, height = faces[0]["box"]
  x1 = abs(x1)
  y1 = abs(y1)

  x2 = x1 + width
  y2 = y1 + height

  face_pixels = pixels[y1:y2, x1:x2]
  img = Image.fromarray(face_pixels)
  img = img.resize(required_size)
  face_array = np.asarray(img)
  return face_array

def load_image(fname):
	img = Image.open(fname)
	img = img.convert("RGB")
	pixels = np.asarray(img)

	return pixels

def load_faces(directory, n):
  model = MTCNN()
  faces = list()
  current_dir = os.getcwd()

  for fname in os.listdir(directory):
    img_path = os.path.sep.join([current_dir, directory, fname])
    pixels = load_image(img_path)

    # Get facial image
    face = extract_face(model, pixels)
    if face is None:
      continue

    faces.append(face)
    if len(faces) >= n:
      break

  return np.asarray(faces)

def plot_faces(faces, n):
  for i in range(n*n):
    plt.subplot(n, n, i+1)
    plt.axis('off')
    plt.imshow(faces[i])
  plt.savefig("artifacts/sample_faces.png")

if __name__ == "__main__":
  dirname = "data/img_align_celeba"

  # faces = load_faces(dirname, 25)
  # print("Loaded faces shape: ", faces.shape)
  # plot_faces(faces, 5)

  n = 50_000
  all_faces = load_faces(dirname, n)
  print("Loaded: ", all_faces.shape)
  np.savez_compressed("img_align_celeba.npz", all_faces)