from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from Model.ops import *

#generator layer blocks
def g_block(inp, style, inoise, fil, u = True):

    if u:
        out = UpSampling2D()(inp)
    else:
        out = Activation('linear')(inp)

    gamma = Dense(fil)(style)
    beta = Dense(fil)(style)

    delta = Lambda(crop_to_fit)([inoise, out])
    delta = Dense(fil, kernel_initializer = 'zeros')(delta)

    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    out = add([out, delta])
    out = Lambda(ada_in)([out, gamma, beta])
    out = LeakyReLU(0.2)(out)

    return out


def make_style_gan_generator(img_dim, latent_size):

  #style network
  style_network = Sequential()

  style_network.add(Dense(512, input_shape = [latent_size]))
  style_network.add(LeakyReLU(0.2))
  style_network.add(Dense(512))
  style_network.add(LeakyReLU(0.2))
  style_network.add(Dense(512))
  style_network.add(LeakyReLU(0.2))
  style_network.add(Dense(512))
  style_network.add(LeakyReLU(0.2))


  #base generator network inputs and architechture

  #define all the inputs
  inp_style = []

  num_layers = int(log2(img_dim) - 1)

  channels = 48

  #create an array for all the style/adain inputs
  for i in range(num_layers):
      inp_style.append(Input([512]))

  #this is the base noise image for the network
  inp_noise = Input([img_dim, img_dim, 1])

  #Latent
  x = Lambda(lambda x: x[:, :128])(inp_style[0])

  #define the base generator network architechture

  x = Dense(4*4*4*channels, activation = 'relu', kernel_initializer = 'he_normal')(x)
  x = Reshape([4, 4, 4*channels])(x)
  #below are the blocks take the three inputs for each layer(previous output, style input [n], the base nois image)
  x = g_block(x, inp_style[0], inp_noise, 16 * channels, u = False)  #4
  x = g_block(x, inp_style[1], inp_noise, 8 * channels)  #8
  x = g_block(x, inp_style[2], inp_noise, 6 * channels)  #16
  x = g_block(x, inp_style[3], inp_noise, 4 * channels)  #32
  x = g_block(x, inp_style[4], inp_noise, 3 * channels)   #64
  x = g_block(x, inp_style[5], inp_noise, 2 * channels)   #128

  if(img_dim == 256):
    x = g_block(x, inp_style[6], inp_noise, 1 * channels)   #256

  x = Conv2D(filters = 3, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(x)

  base_generator = Model(inputs = inp_style + [inp_noise], outputs = x) 

  #now we stitch the style network and generator network together
  inp_style = []
  style = []

  for i in range(num_layers):
      inp_style.append(Input([latent_size]))
      style.append(style_network(inp_style[-1]))

  inp_noise = Input([img_dim, img_dim, 1])

  base_generator_output = base_generator(style + [inp_noise])

  full_generator_model = Model(inputs = inp_style + [inp_noise], outputs = base_generator_output)

  return full_generator_model
  