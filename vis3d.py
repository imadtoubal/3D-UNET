# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib.widgets import Slider, Button
import scipy.io as sio
import numpy as np

from skimage import feature

import tkinter as tk
from tkinter import filedialog

from utils import dice_coef

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

def frame_update(val):
    n = np.floor(slider.val).astype('int') - 1
    p1.set_data(imgseg[:,:,n,:])
    p2.set_data(imsgt[:,:,n,:])

def im2rgb(image):
    image = image.astype('float32')
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype('uint8')
    return np.stack((image, ) * 3, axis=-1)

def applymask(image, seg, intensity=100):
    n = seg.max().astype('int')
    mask = np.zeros_like(image).astype('uint8')
    colors = np.array([
           # k, r, g, b, y, c, m, w
            [0, 1, 0, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 1, 1]
        ])
    for i in range(n):
        color = list(colors[:,i])
        m = (seg == i).astype('uint8')
        mask += (np.stack((m * color[0], m * color[1], m * color[2]), axis=-1)) * intensity

    return image + mask

if __name__ == '__main__':
    matplotlib.use('tkagg')

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    # file_path = 'data/out/out0.mat'
    root.destroy()

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    # ax3 = fig.add_subplot(1, 3, 3)

    plt.subplots_adjust(left=0.1, bottom=0.35)

    img = readmat(file_path, 'data')

    x, y, z, _ = np.shape(img)
    n = np.floor(z/2).astype('int')

    axSlider = plt.axes([0.1, 0.2, 0.8, 0.05])
    slider = Slider(axSlider, 'Slide', valmin=1, valmax=z, valstep=1, valinit=n)

    gt = img[:,:,:,2]
    seg = img[:,:,:,1]
    im = img[:,:,:,0]


    # Draw dice table ============================================================

    row = []
    for j in range(6):
        yt = (gt == j).astype('int')
        yp = (seg == j).astype('int')
        dice = dice_coef(yt, yp, numpy=True) * 100
        row.append(dice)

    axLegend = plt.axes([0.1, 0.15, 0.8, 0.05])
    axLegend.axis('off')
    axLegend.table([
        ['Background', 'Liver', 'Kidney', 'Stomach', 'Duodenum', 'Largebowel', 'Mean'],
        ['{0:.2f}'.format(v) for v in row]+['{0:.2f}'.format(np.sum(row)/len(row))]],
        colLoc='center',
        colColours=['w', 'r', 'g', 'b', 'y', 'c', 'w'])

    #=============================================================================

    imgrgb = im2rgb(im * 0)
    imgseg = applymask(imgrgb, seg)
    imsgt = applymask(imgrgb, gt)

    # ax1.set_title('3D MRI Image')
    ax1.set_title('Segmentation result')
    ax2.set_title('Ground Truth')

    p1 = ax1.imshow(imgseg[:,:,n,:])
    p2 = ax2.imshow(imsgt[:,:,n,:])
    # p3 = ax3.imshow(gt[:,:,n])
    
    

    slider.on_changed(frame_update)
    plt.show()