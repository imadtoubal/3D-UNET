# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib.widgets import Slider, Button
import scipy.io as sio
import numpy as np

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()


fig = plt.figure()
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

plt.subplots_adjust(left=0.1, bottom=0.35)

def readmat(filename, var_name):
    img = sio.loadmat(filename)
    img = img.get(var_name)
    img = img.astype(np.float32)
        
    # unsqueeze for channel size of 1
    return img

img = readmat(file_path, 'data')

x, y, z, _ = np.shape(img)
n = np.floor(z/2).astype('int')

axSlider = plt.axes([0.1, 0.2, 0.8, 0.05])
slider = Slider(axSlider, 'Slide', valmin=1, valmax=z, valstep=1, valinit=n)

gt = img[:,:,:,2]
seg = img[:,:,:,1]
im = img[:,:,:,0]

ax1.set_title('3D MRI Image')
ax2.set_title('Segmentation result')
ax3.set_title('Ground Truth')

p1 = ax1.imshow(im[:,:,n])
p2 = ax2.imshow(seg[:,:,n])
p3 = ax3.imshow(gt[:,:,n])
 
 
def frame_update(val):
    n = np.floor(slider.val).astype('int') - 1
    p1.set_data(im[:,:,n])
    p2.set_data(seg[:,:,n])
    p3.set_data(gt[:,:,n])

slider.on_changed(frame_update)
plt.show()

