from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from Model.ops import *

def make_style_gan_discriminator(img_dim):

  channels = 48

  input = Input(shape = [img_dim, img_dim, 3])

  x = d_block(input, 1 * channels)   #128
  x = d_block(x, 2 * channels)   #64
  x = d_block(x, 3 * channels)   #32
  x = d_block(x, 4 * channels)  #16
  x = d_block(x, 6 * channels)  #8
  x = d_block(x, 8 * channels)  #4
  x = d_block(x, 16 * channels, p = False)  #4

  x = Flatten()(x)

  x = Dense(16 * channels, kernel_initializer = 'he_normal')(x)
  x = LeakyReLU(0.2)(x)

  x = Dense(1, kernel_initializer = 'he_normal')(x)

  discriminator_model = Model(inputs = input, outputs = x)   
  
  return discriminator_model