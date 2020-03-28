# %%
from utils import *
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
import datetime

# Custom imports
from nets import unet, unetpp, scSEunet, unet_bi, unet_2d
from config import get_config

seed(0)
set_random_seed(0)

# %% Import configs
cfg = get_config()


# %%
X,  Y, paths  = load_dataset('data_256_200_64/test/', return_paths=True, pad=28)

# %%
# %%
nets = {
    'unet': unet,
    'unet_bi': unet_bi,
    'unet_2d': unet_2d,
    'scseunet2': scSEunet,
    #'scseunetpp': scSEunetpp,
    # 'attunet': attunet,
    #'unetpp': unetpp,
    #'unetpp_2d': unetpp_2d,
    #'scseunet3': scSEunet
}

for net in nets:
    model = nets[net](256, 256, 64, 1)

    modelname = f'256_model_{net}_fl.p5'
    model.load_weights('ckpt/' + modelname)

    # %%
    out = model.predict(X, batch_size=1)
    if type(out) == type([]):
        out = out[-1]

    export_outs(X, Y, out, cfg['data']['256_out_path'] + modelname +'/', paths=paths)

    tabledata = []

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

    df.to_csv('results/{}'.format(f'256_{modelname}.csv'))



""" model = unet_2d(256, 256, 64, 1)

modelname = '256_model_unet_2d.p5'
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
''' """