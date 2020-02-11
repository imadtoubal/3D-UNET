# %%
import tensorflow as tf 
import scipy.io as sio
import os
import numpy as np
from tqdm import tqdm
import keras
from utils import *
from config import get_config

# %%
print("GPU is available") if tf.test.is_gpu_available() else print("GPU is not available, using CPU")
cfg = get_config()

# %%
# Build the model
def unet(iw, ih, id, ic):
  inputs = tf.keras.layers.Input((iw, ih, id, ic))

  s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
  c1 = tf.keras.layers.Conv3D(16, (3, 3, 3), 
                              activation='relu', 
                              kernel_initializer='he_normal',
                              padding='same')(s)
  c1 = tf.keras.layers.Dropout(0.2)(c1)
  c1 = tf.keras.layers.Conv3D(16, (3, 3, 3), 
                              activation='relu', 
                              kernel_initializer='he_normal',
                              padding='same')(c1)
  p1 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c1)

  # P1  

  c2 = tf.keras.layers.Conv3D(32, (3, 3, 3), 
                              activation='relu', 
                              kernel_initializer='he_normal',
                              padding='same')(p1)
  c2 = tf.keras.layers.Dropout(0.2)(c2)
  c2 = tf.keras.layers.Conv3D(32, (3, 3, 3), 
                              activation='relu', 
                              kernel_initializer='he_normal',
                              padding='same')(c2)
  p2 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c2)

  # P2  

  c3 = tf.keras.layers.Conv3D(64, (3, 3, 3), 
                              activation='relu', 
                              kernel_initializer='he_normal',
                              padding='same')(p2)
  c3 = tf.keras.layers.Dropout(0.2)(c3)
  c3 = tf.keras.layers.Conv3D(64, (3, 3, 3), 
                              activation='relu', 
                              kernel_initializer='he_normal',
                              padding='same')(c3)
  p3 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c3)

  # P3

  c4 = tf.keras.layers.Conv3D(128, (3, 3, 3), 
                              activation='relu', 
                              kernel_initializer='he_normal',
                              padding='same')(p3)
  c4 = tf.keras.layers.Dropout(0.2)(c4)
  c4 = tf.keras.layers.Conv3D(128, (3, 3, 3), 
                              activation='relu', 
                              kernel_initializer='he_normal',
                              padding='same')(c4)
  p4 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c4)

  # P4

  c5 = tf.keras.layers.Conv3D(256, (3, 3, 3), 
                              activation='relu', 
                              kernel_initializer='he_normal',
                              padding='same')(p4)
  c5 = tf.keras.layers.Dropout(0.2)(c5)
  c5 = tf.keras.layers.Conv3D(256, (3, 3, 3), 
                              activation='relu', 
                              kernel_initializer='he_normal',
                              padding='same')(c5)

  # C5

  u6 = tf.keras.layers.Conv3DTranspose(128, (2, 2, 2), 
                                      strides=(2, 2, 2), 
                                      padding='same')(c5)
  u6 = tf.keras.layers.concatenate([u6, c4])
  c6 = tf.keras.layers.Conv3D(128, (3, 3, 3), 
                              activation='relu', 
                              kernel_initializer='he_normal',
                              padding='same')(u6)
  c6 = tf.keras.layers.Dropout(0.2)(c6)
  c6 = tf.keras.layers.Conv3D(128, (3, 3, 3), 
                              activation='relu', 
                              kernel_initializer='he_normal',
                              padding='same')(c6)

  # C6

  u7 = tf.keras.layers.Conv3DTranspose(64, (2, 2, 2), 
                                      strides=(2, 2, 2), 
                                      padding='same')(c6)
  u7 = tf.keras.layers.concatenate([u7, c3])
  c7 = tf.keras.layers.Conv3D(64, (3, 3, 3), 
                              activation='relu', 
                              kernel_initializer='he_normal',
                              padding='same')(u7)
  c7 = tf.keras.layers.Dropout(0.2)(c7)
  c7 = tf.keras.layers.Conv3D(64, (3, 3, 3), 
                              activation='relu', 
                              kernel_initializer='he_normal',
                              padding='same')(c7)

  # C7

  u8 = tf.keras.layers.Conv3DTranspose(32, (2, 2, 2), 
                                      strides=(2, 2, 2), 
                                      padding='same')(c7)
  u8 = tf.keras.layers.concatenate([u8, c2])
  c8 = tf.keras.layers.Conv3D(32, (3, 3, 3), 
                              activation='relu', 
                              kernel_initializer='he_normal',
                              padding='same')(u8)
  c8 = tf.keras.layers.Dropout(0.2)(c8)
  c8 = tf.keras.layers.Conv3D(32, (3, 3, 3), 
                              activation='relu', 
                              kernel_initializer='he_normal',
                              padding='same')(c8)

  # C8

  u9 = tf.keras.layers.Conv3DTranspose(16, (2, 2, 2), 
                                      strides=(2, 2, 2), 
                                      padding='same')(c8)
  u9 = tf.keras.layers.concatenate([u9, c1])
  c9 = tf.keras.layers.Conv3D(16, (3, 3, 3), 
                              activation='relu', 
                              kernel_initializer='he_normal',
                              padding='same')(u9)
  c9 = tf.keras.layers.Dropout(0.2)(c9)
  c9 = tf.keras.layers.Conv3D(16, (3, 3, 3), 
                              activation='relu', 
                              kernel_initializer='he_normal',
                              padding='same')(c9)

  # C9

  outputs = tf.keras.layers.Conv3D(6, (1, 1, 1), activation='softmax')(c9)
  model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
  model.compile(**cfg['net_cmp'])
  return model
