import scipy.io as sio
import os
import numpy as np
from tqdm import tqdm

def readmat(filename, var_name):
    img = sio.loadmat(filename)
    img = img.get(var_name)
    img = img.astype(np.float32)
    
    
    # unsqueeze for channel size of 1
    # return np.expand_dims(img, 0)
    return img

def ind2onehot(indimg):
    indimg = indimg.astype('int')
    classes = indimg.max() + 1
    Y = np.stack([indimg == i for i in range(classes)], axis=4)
    return Y

def load_dataset(root_dir, var_name='data'):
    """
    Args:
        root_dir (string): Directory with all the images.
    """
    # get all .mat files 
    paths = [img_path for img_path in os.listdir(root_dir) if img_path[-4:] == '.mat']
    # read all .mat files
    data = [readmat(root_dir + img_path, var_name) for i, img_path in tqdm(enumerate(paths), total=len(paths))]
    print('Stacking...')
    data = np.stack(data)
    print('Processing...')
    X, Y = data[:,:,:,:,0], data[:,:,:,:,1]
    X = np.expand_dims(X, -1)
    Y = ind2onehot(Y)
    X = np.pad(X, pad_width=((0,0), (14,14), (0,0), (0, 0), (0, 0)), mode='edge')
    Y = np.pad(Y, pad_width=((0,0), (14,14), (0,0), (0, 0), (0, 0)))
    
    return X, Y

# Source: https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a

from keras import backend as K

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)