# Creates GIF animation from the saved plots during training

import imageio
from PIL import Image
import glob

def display_image(epoch_nos):
	img = Image.open("artifacts/image_at_epoch_{:04d}.png".format(epoch_nos))

	return img

def create_anim(filename="dcgan.gif"):
	anim_file = 'dcgan.gif'

	with imageio.get_writer(anim_file, mode='I') as writer:
	  filenames = glob.glob('artifacts/image*.png')
	  filenames = sorted(filenames)
	  last = -1
	  for i,filename in enumerate(filenames):
	    frame = 2*(i**0.5)
	    if round(frame) > round(last):
	      last = frame
	    else:
	      continue
	    image = imageio.imread(filename)
	    writer.append_data(image)
	  image = imageio.imread(anim_file)
	  writer.append_data(image)

if __name__ == "__main__":
	create_anim()