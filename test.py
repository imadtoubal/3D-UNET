from utils import *
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
import datetime

# Custom imports
from unet import unet
from config import get_config

seed(0)
set_random_seed(0)

# %% Import configs
cfg = get_config()


# %%
X,  Y  = load_dataset(cfg['data']['test_path'])

# %%
model = unet(128, 128, 64, 1)

model.load_weights('ckpt/model20200211-181051.p5')


# %%
out = model.predict(X)

export_outs(X, Y, out, cfg['data']['out_path'])