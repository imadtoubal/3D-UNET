from utils import *
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
import datetime
from losses import fl

# Custom imports
from nets import unet, unetpp, unetpp_2d, attunet, scSEunet, scSEunetpp, unet_bi, unet_2d
from config import get_config

#seed(0)
#set_random_seed(0)

# %% Import configs
cfg = get_config()


# %%
X,  Y  = load_dataset(cfg['data']['train_path'])
Xv, Yv = load_dataset(cfg['data']['val_path'])

# %%
nets = {
    #'unet': unet,
    #'unet_bi': unet_bi,
    #'scseunet': scSEunet,
    #'unet_2d': unet_2d,
    #'unetpp': unetpp,
    #'unetpp_2d': unetpp_2d,
    'scseunetpp': scSEunetpp,
    # 'attunet': attunet,
    # 'unetpp_2d': unetpp_2d
}

# %%

loss = fl()
for net in nets:

    print("TRAINING: {} ====================================================".format(net))

    model = nets[net](128, 128, 64, 1, loss=loss)
    model_name = '/model_{}_fl.p5'.format(net)
    checkpointer = tf.keras.callbacks.ModelCheckpoint(cfg['train']['ckpt']['ckpt_path'] + model_name, **cfg['train']['ckpt'])

    callbacks = [
                checkpointer,
                tf.keras.callbacks.EarlyStopping(**cfg['train']['early_stopping']),
                tf.keras.callbacks.TensorBoard(log_dir='logs/fit/{}'.format(model_name), profile_batch=0)
                ]

    Yt  = {'out_{}'.format(o):Y  for o in range(len(model.outputs))}
    Ytv = {'out_{}'.format(o):Yv for o in range(len(model.outputs))}
    
    model.fit(X, Yt,batch_size=1, validation_data=(Xv, Ytv), callbacks=callbacks, **cfg['net'])