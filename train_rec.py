from utils import *
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
import datetime

# Custom imports
from unet import runet
from config import get_config

seed(0)
set_random_seed(0)

# %% Import configs
cfg = get_config()


# %%
X,  Y  = load_dataset(cfg['data']['train_path'])
Xv, Yv = load_dataset(cfg['data']['train_path'])

# %%
model = runet(128, 128, 64, 1, 1)

# %%
model_name = '/model{}.p5'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
checkpointer = tf.keras.callbacks.ModelCheckpoint(cfg['train']['ckpt']['ckpt_path'] + model_name, **cfg['train']['ckpt'])

# %%
callbacks = [
             checkpointer,
             tf.keras.callbacks.EarlyStopping(**cfg['train']['early_stopping']),
             tf.keras.callbacks.TensorBoard(**cfg['train']['tensorboard'])
             ]


# %%
model.fit(X, Y,batch_size=1, validation_data=(Xv, Yv), callbacks=callbacks, **cfg['net'])