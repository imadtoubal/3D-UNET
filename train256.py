from utils import *
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
import datetime

# Custom imports
from nets import unet, unet_2d, unet_bi, unetpp, unetpp_2d, attunet, scSEunet, scSEunetpp
from config import get_config

seed(0)
set_random_seed(0)

# %% Import configs
cfg = get_config()


# %%
X,  Y  = load_dataset(cfg['data']['256_train_path'], pad=28)
Xv, Yv = load_dataset(cfg['data']['256_val_path'], pad=28)

# %%
nets = {
    # 'unet': unet,
    # 'unet_bi': unet_bi,
    # 'unet_2d': unet_2d,
    # 'scseunet': scSEunet,
    # 'unetpp': unetpp,
    # 'scseunetpp': scSEunetpp,
    # 'attunet': attunet,
    # 'unetpp': unetpp,
    'unetpp_2d': unetpp_2d
}


# %%

for net in nets:


    model = nets[net](256, 256, 64, 1)
    # print(net, model.count_params())
    model_name = '/256_model_{}.p5'.format(net)
    checkpointer = tf.keras.callbacks.ModelCheckpoint(cfg['train']['ckpt']['ckpt_path'] + model_name, **cfg['train']['ckpt'])

    callbacks = [
                checkpointer,
                tf.keras.callbacks.EarlyStopping(**cfg['train']['early_stopping']),
                tf.keras.callbacks.TensorBoard(log_dir='logs/fit/{}'.format(model_name), profile_batch=0)
                ]


    Yt  = {'out_{}'.format(o):Y  for o in range(len(model.outputs))}
    Ytv = {'out_{}'.format(o):Yv for o in range(len(model.outputs))}
    model.fit(X, Yt,batch_size=1, validation_data=(Xv, Ytv), callbacks=callbacks, **cfg['net'])