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


class DistributedTrainer:

  def __init__(self,args):

    self.strategy = tf.distribute.MirroredStrategy()
    print ('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))

    self.d_lr = args.d_lr
    self.g_lr = args.g_lr

    self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.d_lr, beta_1=0.0,beta_2=0.9)
    self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.g_lr, beta_1=0.0, beta_2=0.9)

    self.batch_size_per_replica = args.batch_size
    self.global_batch_size = self.batch_size_per_replica * self.strategy.num_replicas_in_sync

    self.num_training_steps = args.num_training_steps

    self.img_dim = args.img_dim

    self.noise_dim = args.noise_dim

    #dataset parameters
    self.data_dir = args.data_dir
    self.dataset = self.init_data_set()

    self.mixed_prob = 0.9

    self.steps_per_epoch = self.files_len / self.global_batch_size
    self.num_epochs = int(self.num_training_steps / self.steps_per_epoch)

    self.gp_weight = np.array([10.0] * self.global_batch_size).astype('float32')

    self.all_disc_loss = []
    self.all_gen_loss = []

    self.print_freq = args.print_freq
    self.save_freq = args.save_freq

  def load_img(self,path):
    img = tf.io.read_file(path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, [self.img_dim, self.img_dim])


  def init_data_set(self):

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
      self.data_dir,
      seed = 123,
      image_size = (self.img_dim,self.img_dim),
      batch_size = self.global_batch_size,
      label_mode = None)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

    dataset = self.strategy.experimental_distribute_dataset(dataset)

    self.files = sorted(glob.glob(self.data_dir + 'images/*')) 
    self.files_len = len(self.files)

    return dataset


  @tf.function
  def train_step(self, discriminator, generator, d_op, g_op, images, style, noise, gp_weights):
    
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


  @tf.function
  def distribute_train_step(self, discriminator, generator, d_op, g_op, images, style, noise, gp_weights):
    per_replica_d_loss, per_replica_g_loss, per_replica_divergence = self.strategy.run(self.train_step, args=(discriminator, generator, d_op, g_op, images, style, noise, gp_weights,))

    total_g_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_g_loss, axis=None)
    total_d_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_d_loss, axis=None)
    total_divergence = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_divergence, axis=None)

    self.all_disc_loss.append(total_g_loss)
    self.all_gen_loss.append(total_d_loss)

    return total_g_loss, total_d_loss, total_divergence



  def train(self):

    #start counter
    self.step_begin_time = time.time()
    self.start_time = time.time()

    #run through all steps using tf dataset

    #calculate epochs based off of desired number of steps
    #this ensures that we can iterate through the tf.dataset correctly

    step_counter = 0

    with self.strategy.scope():

      #setup models
      self.discriminator = make_style_gan_discriminator(self.img_dim)
      self.generator = make_style_gan_generator(self.img_dim,self.noise_dim)

      self.generator.summary()
      self.discriminator.summary()

      for epoch in range(self.num_epochs):

        for batch in self.dataset:

          if(batch.shape[0]!=self.global_batch_size):
            break

          #Train Alternating
          if random.random() < self.mixed_prob:
              style = mixed_list(self.global_batch_size,self.noise_dim,self.img_dim)
          else:
              style = noise_list(self.global_batch_size,self.noise_dim,self.img_dim)

          disc_loss, gen_loss, divergence = self.distribute_train_step(self.discriminator, self.generator, self.discriminator_optimizer,self.generator_optimizer, batch/255., style, noise_image(self.global_batch_size,self.img_dim), self.gp_weight)
          
          self.all_disc_loss.append(disc_loss)
          self.all_gen_loss.append(gen_loss)


          new_weight = 5/(np.array(divergence) + 1e-7)
          self.gp_weight = self.gp_weight[0] * 0.9 + 0.1 * new_weight
          self.gp_weight = np.clip([self.gp_weight] * self.global_batch_size, 0.01, 10000.0).astype('float32')

          #reporting
          if (step_counter % self.print_freq) == 0:

            end_time = time.time()
            diff_time = int(end_time - self.step_begin_time)
            total_time = int(end_time - self.start_time)

            print("Step %d completed. Time took: %s secs. Total time: %s secs" % (step_counter, diff_time, total_time))

            n1 = noise_list(64,self.noise_dim, self.img_dim)
            n2 = noise_image(64, self.img_dim)


            generated_images = self.generator.predict(n1 + [n2], batch_size = self.global_batch_size)

            r = []

            for i in range(0, 64, 8):
                r.append(np.concatenate(generated_images[i:i+8], axis = 1))

            c1 = np.concatenate(r, axis = 0)
            c1 = np.clip(c1, 0.0, 1.0)
            x = Image.fromarray(np.uint8(c1*255.))

            x.save("Results/Images/Distribution/{:06d}.png".format(step_counter))

            # plot_loss(self.all_disc_loss,self.all_gen_loss)

            self.step_begin_time = time.time()

          #save models
          if(step_counter % self.save_freq == 0 and step_counter > 0):

            print("saving model at {}".format(step_counter))

            self.generator.save_weights("SavedModels/generator_weights_at_step_{}.h5".format(step_counter))
            self.discriminator.save_weights("SavedModels/discriminator_weights_at_step_{}.h5".format(step_counter))

          step_counter+=1


