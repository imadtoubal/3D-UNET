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

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()
# file_path = 'data/out/out0.mat'

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
seg = img[:,:,:,1]
im = img[:,:,:,0]

# import plotly.graph_objects as go
X, Y, Z = seg.shape
colors = [
    (203/255, 163/255, 40/255),
    (162/255, 59/255, 114/255),
    (46/255, 134/255, 171/255),
    (199/255, 62/255, 29/255),
    (90/255, 60/255, 75/255)
]

from mayavi import mlab 

for i in range(1, 6):
    x, y, z = np.ogrid[0:X, 0:Y, 0:Z]
    sg = (seg == i).astype('int')
    sg = sg[:,:,::-1]
    s = sg[x,y,z]
    mlab.figure('MRI Segmentation', bgcolor=(1, 1, 1))
    src = mlab.pipeline.scalar_field(s)
    # mlab.pipeline.iso_surface(src, contours=[s.min()+0.1*s.ptp(), ], opacity=1)
    color = colors[i-1]
    mlab.pipeline.iso_surface(src, contours=[s.max()-0.1*s.ptp(), ], color=color)

mlab.show()
