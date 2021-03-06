import argparse
import os
import tensorflow as tf
from Model.generator import *
from Model.ops import *
import cv2
import shutil
import numpy as np
from tqdm import tqdm

def parse_args():
  desc = "create a lerp video"
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('--step', type=int, default=99000, help='The step to load the  model from ')
  parser.add_argument('--num_lerps', type=int, default=10, help='How many lerps to do')
  parser.add_argument('--noise_dim', type=int, default=512, help='The size of the latent vector')
  parser.add_argument('--img_dim', type=int, default=256, help='The dimension of the image')

  return parser.parse_args()

if os.path.exists("Results/GeneratedImages"):
  shutil.rmtree("Results/GeneratedImages")
  os.makedirs('Results/GeneratedImages')
else:
  os.makedirs('Results/GeneratedImages')

args = parse_args()

idx=0

num_lerps = args.num_lerps
noise_dim = args.noise_dim
img_dim = args.img_dim
step = args.step

generator = make_style_gan_generator(img_dim, noise_dim)
generator.load_weights("SavedModels/generator_weights_at_step_{}.h5".format(step))

noise_vector_1 = noise(1,noise_dim)
noise_image_1 = noise_image(1,img_dim)

noise_vector_2 = noise(1,noise_dim)
noise_image_2 = noise_image(1,img_dim)


linX = list(np.linspace(0, 1, 50))

for i in tqdm(range(0,num_lerps)):

  for x in linX:

    frame = None

    #use a linear interpolater 
    lerped_vector = noise_vector_1 * (1-x) + noise_vector_2 * (x)

    noise_vector_list = [lerped_vector] * int(log2(img_dim) - 1)

    lerped_noise_image = noise_image_1 * (1-x) + noise_image_2 * (x)

    image = generator.predict(noise_vector_list + [lerped_noise_image], batch_size = 1)

    resizedImage = cv2.resize(image[0]*255., dsize=(512, 512), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite('Results/GeneratedImages/image{}.png'.format('%04d'%idx), cv2.cvtColor(resizedImage, cv2.COLOR_RGB2BGR))

    idx+=1

  noise_vector_1 = noise_vector_2
  noise_image_1 = noise_image_2

  noise_vector_2 = noise(1,noise_dim)
  noise_image_2 = noise_image(1,img_dim)
