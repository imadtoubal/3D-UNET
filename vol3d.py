# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import matplotlib
matplotlib.use('tkagg')
import scipy.io as sio
import numpy as np

from tqdm import tqdm

from skimage import feature

import tkinter as tk
from tkinter import filedialog

# root = tk.Tk()
# root.withdraw()

# file_path = filedialog.askopenfilename()
file_path = 'data/out/out0.mat'

def readmat(filename, var_name):
    img = sio.loadmat(filename)
    img = img.get(var_name)
    img = img.astype(np.float32)
        
    # unsqueeze for channel size of 1
    return img

def canny3d(seg):
    classes = (seg.max() + 1).astype('int')
    out = np.zeros_like(seg)
    for i in range(classes):
        seg_c = seg == i
        for z in range(seg_c.shape[2]):
            out[:,:,z] += feature.canny(seg_c[:,:,z]) * i
    
    return out

img = readmat(file_path, 'data')

gt = img[:,:,:,2]
seg = canny3d(img[:,:,:,1])
im = img[:,:,:,0]

import plotly.graph_objects as go
x, y, z = seg.shape

X, Y, Z = np.mgrid[0:x:2, 0:y:2, 0:z:2]
values = seg[X, Y, Z]
print(X.shape, Y.shape, Z.shape)
fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    isomin=0.1,
    isomax=0.8,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=17, # needs to be a large number for good volume rendering
    ))
fig.show()
