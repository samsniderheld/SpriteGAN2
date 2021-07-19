from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from Model.ops import *

def make_style_gan_discriminator(img_dim):

  # channels = 48

  input = Input(shape = [img_dim, img_dim, 3])


  x = d_block(input, 32)   #256
  x = d_block(x, 64)   #128
  x = d_block(x, 128)   #64
  x = d_block(x, 256)  #32
  x = d_block(x, 512)  #16

  if(img_dim == 256):
    x = d_block(x,512,  p = False)

  if(img_dim == 512):
    x = d_block(x, 512) #8
    x = d_block(x, 512, p = False)  #4    

  x = Flatten()(x)

  x = Dense(32, kernel_initializer = 'he_normal')(x)
  x = LeakyReLU(0.2)(x)

  x = Dense(1, kernel_initializer = 'he_normal')(x)

  discriminator_model = Model(inputs = input, outputs = x)   
  
  return discriminator_model