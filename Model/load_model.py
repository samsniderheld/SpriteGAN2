from Model.discriminator import make_style_gan_discriminator
from Model.generator import make_style_gan_generator
import tensorflow as tf

def load_models_from_step(args):

	discriminator = make_style_gan_discriminator(args.img_dim)
	generator = make_style_gan_generator(args.img_dim,args.noise_dim)

	discriminator.load_weights("SavedModels/discriminator_weights_at_step_{}.h5".format(args.step))
	generator.load_weights("SavedModels/generator_weights_at_step_{}.h5".format(args.step))

	return discriminator, generator


def load_generator_from_step(args):
	
	generator = make_style_gan_generator(args.img_dim,args.noise_dim)

	generator.load_weights("SavedModels/generator_weights_at_step_{}.h5".format(args.step))

	return generator