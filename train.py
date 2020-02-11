from utils import *
import numpy as np
import tensorflow as tf

# Custom imports
from unet import unet
from config import get_config

# %% Import configs
cfg = get_config()


# %%
X,  Y  = load_dataset(cfg['data']['train_path'])
Xv, Yv = load_dataset(cfg['data']['train_path'])

# %%
model = unet(128, 128, 64, 1)

# %%

checkpointer = tf.keras.callbacks.ModelCheckpoint(cfg['train']['ckpt']['ckpt_path'] + '/model.h5', **cfg['train']['ckpt'])

# %%
callbacks = [
             checkpointer,
             tf.keras.callbacks.EarlyStopping(**cfg['train']['early_stopping']),
             tf.keras.callbacks.TensorBoard(**cfg['train']['tensorboard'])
             ]


# %%
model.fit(X, Y,batch_size=1, validation_data=(Xv, Yv), callbacks=callbacks, **cfg['net'])