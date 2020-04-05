from utils import *
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
import datetime
from losses import fl

# Custom imports
from nets import unet, unet_mini, unetpp, unetpp_2d, attunet, scSEunet, scSEunetpp, unet_bi, unet_2d
from config import get_config

#seed(0)
#set_random_seed(0)

# %% Import configs
cfg = get_config()


# %%
X,  Y  = np.load('patches/train_train/x_patches.npy'), np.load('patches/train_train/y_patches.npy')
Xv, Yv = np.load('patches/train_val/x_patches.npy'), np.load('patches/train_val/y_patches.npy')

# %%
nets = {
    #'0.unet': unet_mini,
    '1.unet': unet,
    #'unet_bi': unet_bi,
    #'3.scseunet': scSEunet,
    #'2.unet_2d': unet_2d,
    #'4.unetpp': unetpp,
    #'5.unetpp_2d': unetpp_2d,
    #'6.scseunetpp': scSEunetpp,
    # 'attunet': attunet,
    # 'unetpp_2d': unetpp_2d
}

# %%
for net in nets:

    print("TRAINING: {} ====================================================".format(net))

    model = nets[net](64, 64, 16, 10, bn=True)
    model_name = '/model_{}_patches.p5'.format(net)
    checkpointer = tf.keras.callbacks.ModelCheckpoint('ckpt_patches' + model_name, **cfg['train']['ckpt'])

    callbacks = [
                checkpointer,
                tf.keras.callbacks.EarlyStopping(**cfg['train']['early_stopping']),
                tf.keras.callbacks.TensorBoard(log_dir='logs_patches/fit/{}'.format(model_name), profile_batch=0)
                ]

    Yt  = {'out_{}'.format(o):Y  for o in range(len(model.outputs))}
    Ytv = {'out_{}'.format(o):Yv for o in range(len(model.outputs))}
    
    model.fit(X, Yt,batch_size=16, validation_data=(Xv, Ytv), callbacks=callbacks, **cfg['net'])