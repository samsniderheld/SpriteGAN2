from Model.generator import *
from Model.discriminator import *
from Training.loss_functions import *
from Utils.reporting import *
from Model.ops import *
import tensorflow as tf
import numpy as np
import time
import glob
import random
from PIL import Image



def train(args):

  d_lr = args.d_lr
  g_lr = args.g_lr

  #setup optimizers
  discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=d_lr, beta_1=0.0,beta_2=0.9)
  generator_optimizer = tf.keras.optimizers.Adam(learning_rate=g_lr, beta_1=0.0, beta_2=0.9)

  #setup tf.dataset
  data_dir = args.data_dir
  
  dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    seed = 123,
    image_size = (args.img_dim,args.img_dim),
    batch_size = args.batch_size,
    label_mode = None)

  #based model seems to normalize between 0-1 not -1 and 1
  # normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255.)

  files = sorted(glob.glob(data_dir + 'images/*')) 
  files_len = len(files)

  mixed_prob = 0.9

  #setup models
  discriminator = make_style_gan_discriminator(args.img_dim)
  generator = make_style_gan_generator(args.img_dim,args.noise_dim)


  generator.summary()
  discriminator.summary()

  gp_weight = np.array([10.0] * args.batch_size).astype('float32')

  #setup reporting lists
  all_disc_loss = []
  all_gen_loss = []

  #start counter
  step_begin_time = time.time()
  start_time = time.time()


  #run through all steps using tf dataset

  #calculate epochs based off of desired number of steps
  #this ensures that we can iterate through the tf.dataset correctly

  steps_per_epoch = files_len / args.batch_size
  num_epochs = int(args.num_training_steps / steps_per_epoch)

  step_counter = 0

  for epoch in range(num_epochs):

    for batch in dataset:

      if(batch.shape[0]!=args.batch_size):
        break

      #Train Alternating
      if random.random() < mixed_prob:
          style = mixed_list(args.batch_size,args.noise_dim,args.img_dim)
      else:
          style = noise_list(args.batch_size,args.noise_dim,args.img_dim)

      disc_loss, gen_loss, divergence = train_step(discriminator, generator, discriminator_optimizer,generator_optimizer, batch/255., style, noise_image(args.batch_size,args.img_dim), gp_weight)
      
      all_disc_loss.append(disc_loss)
      all_gen_loss.append(gen_loss)


      new_weight = 5/(np.array(divergence) + 1e-7)
      gp_weight = gp_weight[0] * 0.9 + 0.1 * new_weight
      gp_weight = np.clip([gp_weight] * args.batch_size, 0.01, 10000.0).astype('float32')

      #reporting
      if (step_counter % args.print_freq) == 0:

        end_time = time.time()
        diff_time = int(end_time - step_begin_time)
        total_time = int(end_time - start_time)

        print("Step %d completed. Time took: %s secs. Total time: %s secs" % (step_counter, diff_time, total_time))

        n1 = noise_list(64,args.noise_dim, args.img_dim)
        n2 = noise_image(64, args.img_dim)


        generated_images = generator.predict(n1 + [n2], batch_size = args.batch_size)

        r = []

        for i in range(0, 64, 8):
            r.append(np.concatenate(generated_images[i:i+8], axis = 1))

        c1 = np.concatenate(r, axis = 0)
        c1 = np.clip(c1, 0.0, 1.0)
        x = Image.fromarray(np.uint8(c1*255.))
        x = x.resize(1024,1024)
        
        x.save("Results/Images/Distribution/{:06d}.png".format(step_counter))

        # plot_loss(all_disc_loss,all_gen_loss)

        step_begin_time = time.time()

      #save models
      if(step_counter % args.save_freq == 0 and step_counter > 0):

        print("saving model at {}".format(step_counter))

        generator.save_weights("SavedModels/generator_weights_at_step_{}.h5".format(step_counter))
        discriminator.save_weights("SavedModels/discriminator_weights_at_step_{}.h5".format(step_counter))

      step_counter+=1

@tf.function
def train_step(discriminator, generator, d_op, g_op, images, style, noise, gp_weights):
  
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(style + [noise], training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = K.mean(fake_output)
    divergence = K.mean(K.relu(1 + real_output) + K.relu(1 - fake_output))
    disc_loss = divergence + gradient_penalty(images, real_output, gp_weights)

  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  g_op.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  d_op.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

  return disc_loss, gen_loss, divergence

