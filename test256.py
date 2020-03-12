# %%
from utils import *
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
import datetime

# Custom imports
from nets import unet, unetpp, scSEunet
from config import get_config

seed(0)
set_random_seed(0)

# %% Import configs
cfg = get_config()


# %%
X,  Y, paths  = load_dataset('data_256_200_64/test/', return_paths=True)

# %%
model = scSEunet(256, 256, 64, 1)

modelname = '256_model_scseunet_20200228-182134.p5'
model.load_weights('ckpt/' + modelname)


# %%
out = model.predict(X, batch_size=1)
export_outs(X, Y, out, cfg['data']['256_out_path'] + modelname + '/', paths)

tabledata = []

# %%

for i in range(out.shape[0]):
    row = [paths[i]]
    for j in range(1, 6):
        yt = Y[i,:,:,:,j]
        yp = out[i,:,:,:,j]
        dice = dice_coef(yt, yp, numpy=True) * 100
        row.append('%.2f' % dice)
    tabledata.append(row)

import pandas as pd 

df = pd.DataFrame(tabledata)

print(df)

outcsv = '{}{}.csv'.format(cfg['data']['256_out_path'], modelname)
df.to_csv(outcsv)

# %%
