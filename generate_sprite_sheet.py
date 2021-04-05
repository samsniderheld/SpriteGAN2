import argparse
import os
import tensorflow as tf
from Model.generator import *
from Model.ops import *
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image

def parse_args():
  desc = "create a lerp video"
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('--step', type=int, default=10000, help='The step to load the  model from ')
  parser.add_argument('--noise_dim', type=int, default=512, help='The size of the latent vector')
  parser.add_argument('--img_dim', type=int, default=128, help='The dimension of the image')
  parser.add_argument('--output_sprite_dim', type=int, default=128, help='The dimension of each sprite in the sheet')
  parser.add_argument('--sprite_sheet_dim', type=int, default=10, help='The Sprite Sheet Dim')

  return parser.parse_args()

if os.path.exists("Results/SpriteSheets"):
  shutil.rmtree("Results/SpriteSheets")
  os.makedirs('Results/SpriteSheets')
else:
  os.makedirs('Results/SpriteSheets')

args = parse_args()

idx=0

step = args.step
noise_dim = args.noise_dim
sprite_sheet_dim = args.sprite_sheet_dim
output_sprite_dim = args.output_sprite_dim
img_dim = args.img_dim
imgs = []


generator = make_style_gan_generator(img_dim, noise_dim)

generator.load_weights("SavedModels/generator_weights_at_step_{}.h5".format(step))


v1 = noise(1,args.noise_dim)
v2 = noise(1,args.noise_dim)

noise_image = noise_image(1,args.img_dim)

linX = list(np.linspace(0, 1, sprite_sheet_dim))

startV = v1

for i in tqdm(range(0,sprite_sheet_dim)):

  for x in linX:

    frame = None

    v = [v1 * (1-x) + v2 * (x)] * int(log2(img_dim) - 1)

    #get the output and reshape it 
    y = generator.predict(v + [noise_image], batch_size = 1)

    y = cv2.resize(y[0]*255., dsize=(output_sprite_dim, output_sprite_dim), interpolation=cv2.INTER_NEAREST)

    imgs.append(y)

    idx+=1

  if i  == sprite_sheet_dim - 2 :
    v1 = v2
    v2 = startV
  else:
    v1 = v2
    v2 = noise(1,512)

v_stacks = []

for i in range(0,pow(sprite_sheet_dim,2),sprite_sheet_dim):
  v_stacks.append(cv2.hconcat(imgs[i:i+sprite_sheet_dim]))

full_img = cv2.vconcat(v_stacks)

cv2.imwrite('Results/SpriteSheets/SpriteSheet.jpg', cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR))