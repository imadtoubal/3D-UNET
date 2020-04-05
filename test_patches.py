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
X,  Y  = np.load('patches/test/x_patches.npy'), np.load('patches/test/y_patches.npy')

# %%
nets = {
    '0.unet': unet_mini,
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
    model = nets[net](64,64,16,10, bn=True)
    modelname = f'model_{net}_patches.p5'

    model.load_weights('ckpt_patches/' + modelname)

    # %%
    out = model.predict(X, batch_size=1)
    if type(out) == type([]):
        out = out[-1]

    tabledata = []

    for i in range(out.shape[0]):
        row = [f'{i}']
        for j in range(1, 6):
            yt = Y[i,:,:,:,j]
            yp = out[i,:,:,:,j]
            dice = dice_coef(yt, yp, numpy=True) * 100
            row.append('%.2f' % dice)
        tabledata.append(row)

    import pandas as pd 

    df = pd.DataFrame(tabledata)

    print(df)

    df.to_csv('results/{}'.format(f'{modelname}_patches.csv'))
