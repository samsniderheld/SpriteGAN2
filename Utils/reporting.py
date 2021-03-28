import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np



def plot_loss(all_disc_loss,all_gen_loss):

  plt.figure(figsize=(10,5))
  plt.plot(np.arange(len(all_disc_loss)),all_disc_loss,label='D')
  plt.plot(np.arange(len(all_gen_loss)),all_gen_loss,label='G')
  plt.legend()
  plt.title('All Time Loss')
  plt.savefig('Results/Images/Loss/all_losses')
  plt.show()
  
  plt.close('all')



