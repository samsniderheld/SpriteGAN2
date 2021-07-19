from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from math import floor, log2
import random



def noise(n,latent_size):
    return np.random.normal(0.0, 1.0, size = [n, latent_size]).astype('float32')

def noise_list(n,latent_size,img_dim):
    n_layers = int(log2(img_dim) - 1)
    return [noise(n,latent_size)] * n_layers

def mixed_list(n,latent_size,img_dim):
    n_layers = int(log2(img_dim) - 1)
    tt = int(random.random() * n_layers)
    p1 = [noise(n,latent_size)] * tt
    p2 = [noise(n,latent_size)] * (n_layers - tt)
    return p1 + [] + p2

def noise_image(n,img_dim):
    return np.random.uniform(0.0, 1.0, size = [n, img_dim, img_dim, 1]).astype('float32')


#lambda functions
def ada_in(x):
    #Normalize x[0]
    mean = K.mean(x[0], axis = [1, 2], keepdims = True)
    std = K.std(x[0], axis = [1, 2], keepdims = True) + 1e-7
    y = (x[0] - mean) / std

    #Reshape gamma and beta
    pool_shape = [-1, 1, 1, y.shape[-1]]
    g = tf.reshape(x[1], pool_shape) + 1.0
    b = tf.reshape(x[2], pool_shape)

    #Multiply by x[1] (GAMMA) and add x[2] (BETA)
    return y * g + b

#takes previous layer output, and input noise image and crops the layer output to fit the noise image dims
def crop_to_fit(x):
    # x[0] - image noise
    # x[1] - previous layer output

    height = x[1].shape[1]
    width = x[1].shape[2]

    return x[0][:, :height, :width, :]



def d_block(inp, fil, p = True):

    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(inp)
    out = LeakyReLU(0.2)(out)

    if p:
        out = AveragePooling2D()(out)

    return out


