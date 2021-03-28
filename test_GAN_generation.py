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
import matplotlib.pyplot as plt
import glob
from sklearn.decomposition import PCA
from scipy.spatial import distance

if os.path.exists("Results/TestGeneration"):
  shutil.rmtree("Results/TestGeneration")
  os.makedirs('Results/TestGeneration')
else:
  os.makedirs('Results/TestGeneration')

def parse_args():
  desc = "tests the models generative capability"
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('--data_dir', type=str, default="Data/SaganRaw/", help='The directory that holds the image data')
  parser.add_argument('--step', type=int, default=99000, help='The step to load the  model from ')
  parser.add_argument('--noise_dim', type=int, default=512, help='The size of the latent vector')
  parser.add_argument('--img_dim', type=int, default=256, help='The dimension of the image')
  parser.add_argument('--num_imgs_to_test', type=int, default=10, help='How many images are we going to test')
  parser.add_argument('--data_step', type=int, default=1, help='every nth data sample to skip')


  return parser.parse_args()

def load_img(path):
    img = tf.io.read_file(path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, [224, 224])/255

args = parse_args()

data_dir = args.data_dir
noise_dim = args.noise_dim
img_dim = args.img_dim
data_step = args.data_step

#load our model from arguments

generator = make_style_gan_generator(img_dim, noise_dim)

generator.load_weights("SavedModels/generator_weights_at_step_{}.h5".format(args.step))

numImages = args.num_imgs_to_test
dataset = []

#generate a set of random images from our model.
#these will be the images we test against the training dataset.

print("Generate Test Images")

for i in tqdm(range(numImages)):
  noiseVectorList = [noise(1,512)] * int(log2(256) - 1)
  noiseImage = noise_image(1,256)
  image = generator.predict(noiseVectorList + [noiseImage], batch_size = 1)
  image = cv2.resize(image[0], dsize=(224, 224), interpolation=cv2.INTER_NEAREST)
  dataset.append(image)


#load in VGG model and cut off the last layer to get our feature extractor
vgg = tf.keras.applications.VGG16()
featureExtractor = tf.keras.Model(vgg.input,vgg.get_layer("fc2").output)

print("Feature extractor created")

dataInput = sorted(glob.glob(data_dir + "images/*")) 

print("Testing against "+ str(len(dataInput)) + " images.")

print("Converting to dataset")

for i in tqdm(range(0,len(dataInput),data_step)):
  newImage = load_img(dataInput[i])
  dataset.append(newImage)

print("converting dataset into features with VGG16")

features = []
idx = 0
for i in tqdm(range(0,len(dataset))):
  sample = np.array(dataset[i])
  sample = np.expand_dims(sample,axis=0)
  feature = featureExtractor(sample)
  features.append(feature[0])

features = np.array(features)

pca = PCA(n_components=300)
pca.fit(features)

print("Performing Principal Component Analysis on test images and dataset")

pca_features = pca.transform(features)

rows = []

for i in tqdm(range(0,numImages)):

  similar_idx = [ distance.cosine(pca_features[i], feat) for feat in pca_features[numImages:]]

  idx_closest = sorted(range(len(similar_idx)), key=lambda k: similar_idx[k])[0:numImages-1]

  # load all the similarity results as thumbnails of height 100
  thumbs = []
  for idx in idx_closest:
      thumbs.append(dataset[idx])
  thumbs.insert(0,dataset[i])
  # concatenate the images into a single image
  concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)

  rows.append(concat_image)


final_image = np.concatenate([np.asarray(t) for t in rows], axis=0)

plt.figure(figsize = (20,20))
plt.imshow(final_image)

plt.savefig("Results/TestGeneration/Test.png")


