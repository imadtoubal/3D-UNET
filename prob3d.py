# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib.widgets import Slider, Button
import scipy.io as sio
import numpy as np

from skimage import feature

import tkinter as tk
from tkinter import filedialog

from utils import dice_coef

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()
root.destroy()


fig = plt.figure()
axes = [fig.add_subplot(i) for i in range(231,237)]


plt.subplots_adjust(left=0.1, bottom=0.35)

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

x, y, z, c = np.shape(img)
n = np.floor(z/2).astype('int')

axSlider = plt.axes([0.1, 0.2, 0.9, 0.05])
slider = Slider(axSlider, 'Slide', valmin=1, valmax=z, valstep=1, valinit=n)

p = []
for i in range(6):
    p.append(axes[i].imshow(img[:,:,n,i]))
    
 
 
def frame_update(val):
    n = np.floor(slider.val).astype('int') - 1
    for i in range(6):
        p[i].set_data(img[:,:,n,i])
    

slider.on_changed(frame_update)
plt.show()

