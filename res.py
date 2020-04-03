# %%
from utils import *
import numpy as np
from config import get_config
from scipy.ndimage import zoom
from skimage.transform import rescale, downscale_local_mean

# %% Import configs

cfg = get_config()

# %%
X,  Y  = load_dataset(cfg['data']['256_train_path'], pad=28)
Xv, Yv = load_dataset(cfg['data']['256_val_path'], pad=28)
Xt, Yt = load_dataset(cfg['data']['256_test_path'], pad=28)

# %%
ores = 256

def resize(x, resl):
    shape = [int(s * resl) for s in x.shape]
    shape[0] = x.shape[0]
    shape[-1] = x.shape[-1]

    output = np.zeros(shape)
    for j in range(x.shape[0]):
        output[j] = rescale(x[j], resl, multichannel=True)

    return output

# %%
for i in range(0, 5):
    for split, x, y in zip(('train', 'val', 'test'), (X, Xv, Xt), (Y, Yv, Yt)):
        # resze x by 1/2^i
        resl = 1/2**i
        res = int(np.floor(256 * resl))
        xname = f'data_npy/{res}_x_{split}.npy'
        yname = f'data_npy/{res}_y_{split}.npy'
        x = resize(x, resl)
        y = resize(y, resl)
        np.save(xname, x)
        np.save(yname, y)



# %%
