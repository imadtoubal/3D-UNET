# %%
import tensorflow as tf 
import scipy.io as sio
import os
import numpy as np
from tqdm import tqdm
import keras
from utils import *
from config import get_config

import keras.backend as K

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


def runet(iw, ih, id, ic, rec=1):
  inp = tf.keras.layers.Input((iw, ih, id, ic))
  z = inp * 0
  padded_inp = tf.keras.layers.concatenate([inp, z, z, z, z, z, z])

  inputs = [padded_inp]
  
  # Block 1
  conv1_1 = tf.keras.layers.Conv3D(16, (3, 3, 3), 
                                activation='relu', 
                                kernel_initializer='he_normal',
                                padding='same')
  conv1_2 = tf.keras.layers.Conv3D(16, (3, 3, 3), 
                                activation='relu', 
                                kernel_initializer='he_normal',
                                padding='same')
  maxpool1 = tf.keras.layers.MaxPooling3D((2, 2, 2))

  # Block 2
  conv2_1 = tf.keras.layers.Conv3D(32, (3, 3, 3), 
                                activation='relu', 
                                kernel_initializer='he_normal',
                                padding='same')
  conv2_2 = tf.keras.layers.Conv3D(32, (3, 3, 3), 
                                activation='relu', 
                                kernel_initializer='he_normal',
                                padding='same')
  maxpool2 = tf.keras.layers.MaxPooling3D((2, 2, 2))

  # Block 3
  conv3_1 = tf.keras.layers.Conv3D(64, (3, 3, 3), 
                                activation='relu', 
                                kernel_initializer='he_normal',
                                padding='same')
  conv3_2 = tf.keras.layers.Conv3D(64, (3, 3, 3), 
                                activation='relu', 
                                kernel_initializer='he_normal',
                                padding='same')
  maxpool3 = tf.keras.layers.MaxPooling3D((2, 2, 2))
  
  # Block 4 
  conv4_1 = tf.keras.layers.Conv3D(128, (3, 3, 3), 
                                activation='relu', 
                                kernel_initializer='he_normal',
                                padding='same')
  conv4_2 = tf.keras.layers.Conv3D(128, (3, 3, 3), 
                                activation='relu', 
                                kernel_initializer='he_normal',
                                padding='same')
  maxpool4 = tf.keras.layers.MaxPooling3D((2, 2, 2))
  
  # Block 5 
  conv5_1 = tf.keras.layers.Conv3D(256, (3, 3, 3), 
                                activation='relu', 
                                kernel_initializer='he_normal',
                                padding='same')
  dropout5 =  tf.keras.layers.Dropout(0.2)
  conv5_2 = tf.keras.layers.Conv3D(256, (3, 3, 3), 
                                activation='relu', 
                                kernel_initializer='he_normal',
                                padding='same')

  # Block 6
  transconv6 = tf.keras.layers.Conv3DTranspose(128, (2, 2, 2), 
                                        strides=(2, 2, 2), 
                                        padding='same')
  conv6_1 = tf.keras.layers.Conv3D(128, (3, 3, 3), 
                                activation='relu', 
                                kernel_initializer='he_normal',
                                padding='same')
  conv6_2 = tf.keras.layers.Conv3D(128, (3, 3, 3), 
                                activation='relu', 
                                kernel_initializer='he_normal',
                                padding='same')

  # Block 7
  transconv7 = tf.keras.layers.Conv3DTranspose(64, (2, 2, 2), 
                                        strides=(2, 2, 2), 
                                        padding='same')
  conv7_1 = tf.keras.layers.Conv3D(64, (3, 3, 3), 
                                activation='relu', 
                                kernel_initializer='he_normal',
                                padding='same')
  conv7_2 = tf.keras.layers.Conv3D(64, (3, 3, 3), 
                                activation='relu', 
                                kernel_initializer='he_normal',
                                padding='same')

  # Block 8
  transconv8 = tf.keras.layers.Conv3DTranspose(32, (2, 2, 2), 
                                        strides=(2, 2, 2), 
                                        padding='same')
  conv8_1 = tf.keras.layers.Conv3D(32, (3, 3, 3), 
                                activation='relu', 
                                kernel_initializer='he_normal',
                                padding='same')
  conv8_2 = tf.keras.layers.Conv3D(32, (3, 3, 3), 
                                activation='relu', 
                                kernel_initializer='he_normal',
                                padding='same')

  # Block 9
  transconv9 = tf.keras.layers.Conv3DTranspose(16, (2, 2, 2), 
                                        strides=(2, 2, 2), 
                                        padding='same')
  conv9_1 = tf.keras.layers.Conv3D(16, (3, 3, 3), 
                                activation='relu', 
                                kernel_initializer='he_normal',
                                padding='same')
  conv9_2 = tf.keras.layers.Conv3D(16, (3, 3, 3), 
                                activation='relu', 
                                kernel_initializer='he_normal',
                                padding='same')

  # Output 
  outconv = tf.keras.layers.Conv3D(6, (1, 1, 1), activation='softmax')

  for i in range(rec):

    c1 = conv1_1(inputs[i])
    c1 = tf.keras.layers.Dropout(0.2)(c1)
    c1 = conv1_2(c1)
    p1 = maxpool1(c1)

    # P1  

    c2 = conv2_1(p1)
    c2 = tf.keras.layers.Dropout(0.2)(c2)
    c2 = conv2_2(c2)
    p2 = maxpool2(c2)

    # P2  

    c3 = conv3_1(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = conv3_2(c3)
    p3 = maxpool3(c3)

    # P3

    c4 = conv4_1(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = conv4_2(c4)
    p4 = maxpool4(c4)

    # P4

    c5 = conv5_1(p4)
    c5 = dropout5(c5)
    c5 = conv5_2(c5)

    # C5

    u6 = transconv6(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = conv6_1(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = conv6_2(c6)

    # C6

    u7 = transconv7(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = conv7_1(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = conv7_2(c7)
    

    # C7

    u8 = transconv8(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = conv8_1(u8)
    c8 = tf.keras.layers.Dropout(0.2)(c8)
    c8 = conv8_2(c8)

    # C8

    u9 = transconv9(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = conv9_1(u9)
    c9 = tf.keras.layers.Dropout(0.2)(c9)
    c9 = conv9_2(c9)

    # C9  
    outputs = outconv(c9)

    nextinp = tf.keras.layers.Conv3D(6, (3, 3, 3), trainable=False, padding='same', 
                                      kernel_initializer='ones', bias_initializer='zeros')(outputs)
    nextinp = tf.keras.layers.concatenate([nextinp, inp])
    inputs.append(nextinp)

  model = tf.keras.Model(inputs=[inp], outputs=[outputs])
  model.compile(**cfg['net_cmp'])
  return model

def dunet(iw, ih, id, ic, rec=1):
  inp = tf.keras.layers.Input((iw, ih, id, ic))
  z = inp * 0
  padded_inp = tf.keras.layers.concatenate([inp, z, z, z, z, z, z])

  inputs = [padded_inp]
  conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, conv4_1, conv4_2, conv5_1, conv5_2 = [], [], [], [], [], [], [], [], [], []
  conv6_1, conv6_2, conv7_1, conv7_2, conv8_1, conv8_2, conv9_1, conv9_2 = [], [], [], [], [], [], [], []
  maxpool1, maxpool2, maxpool3, maxpool4, maxpool5 = [], [], [], [], []
  transconv6, transconv7, transconv8, transconv9 = [], [], [], []
  outconv = []

  for i in range(rec):
    # Block 1
    conv1_1.append(tf.keras.layers.Conv3D(16, (3, 3, 3), 
                                  activation='relu', 
                                  kernel_initializer='he_normal',
                                  padding='same'))
    conv1_2.append(tf.keras.layers.Conv3D(16, (3, 3, 3), 
                                  activation='relu', 
                                  kernel_initializer='he_normal',
                                  padding='same'))
    maxpool1.append(tf.keras.layers.MaxPooling3D((2, 2, 2)))

    # Block 2
    conv2_1.append(tf.keras.layers.Conv3D(32, (3, 3, 3), 
                                  activation='relu', 
                                  kernel_initializer='he_normal',
                                  padding='same'))
    conv2_2.append(tf.keras.layers.Conv3D(32, (3, 3, 3), 
                                  activation='relu', 
                                  kernel_initializer='he_normal',
                                  padding='same'))
    maxpool2.append(tf.keras.layers.MaxPooling3D((2, 2, 2)))

    # Block 3
    conv3_1.append(tf.keras.layers.Conv3D(64, (3, 3, 3), 
                                  activation='relu', 
                                  kernel_initializer='he_normal',
                                  padding='same'))
    conv3_2.append(tf.keras.layers.Conv3D(64, (3, 3, 3), 
                                  activation='relu', 
                                  kernel_initializer='he_normal',
                                  padding='same'))
    maxpool3.append(tf.keras.layers.MaxPooling3D((2, 2, 2)))
    
    # Block 4 
    conv4_1.append(tf.keras.layers.Conv3D(128, (3, 3, 3), 
                                  activation='relu', 
                                  kernel_initializer='he_normal',
                                  padding='same'))
    conv4_2.append(tf.keras.layers.Conv3D(128, (3, 3, 3), 
                                  activation='relu', 
                                  kernel_initializer='he_normal',
                                  padding='same'))
    maxpool4.append(tf.keras.layers.MaxPooling3D((2, 2, 2)))
    
    # Block 5 
    conv5_1.append(tf.keras.layers.Conv3D(256, (3, 3, 3), 
                                  activation='relu', 
                                  kernel_initializer='he_normal',
                                  padding='same'))

    conv5_2.append(tf.keras.layers.Conv3D(256, (3, 3, 3), 
                                  activation='relu', 
                                  kernel_initializer='he_normal',
                                  padding='same'))

    # Block 6
    transconv6.append(tf.keras.layers.Conv3DTranspose(128, (2, 2, 2), 
                                          strides=(2, 2, 2), 
                                          padding='same'))
    conv6_1.append(tf.keras.layers.Conv3D(128, (3, 3, 3), 
                                  activation='relu', 
                                  kernel_initializer='he_normal',
                                  padding='same'))
    conv6_2.append(tf.keras.layers.Conv3D(128, (3, 3, 3), 
                                  activation='relu', 
                                  kernel_initializer='he_normal',
                                  padding='same'))

    # Block 7
    transconv7.append(tf.keras.layers.Conv3DTranspose(64, (2, 2, 2), 
                                          strides=(2, 2, 2), 
                                          padding='same'))
    conv7_1.append(tf.keras.layers.Conv3D(64, (3, 3, 3), 
                                  activation='relu', 
                                  kernel_initializer='he_normal',
                                  padding='same'))
    conv7_2.append(tf.keras.layers.Conv3D(64, (3, 3, 3), 
                                  activation='relu', 
                                  kernel_initializer='he_normal',
                                  padding='same'))

    # Block 8
    transconv8.append(tf.keras.layers.Conv3DTranspose(32, (2, 2, 2), 
                                          strides=(2, 2, 2), 
                                          padding='same'))
    conv8_1.append(tf.keras.layers.Conv3D(32, (3, 3, 3), 
                                  activation='relu', 
                                  kernel_initializer='he_normal',
                                  padding='same'))
    conv8_2.append(tf.keras.layers.Conv3D(32, (3, 3, 3), 
                                  activation='relu', 
                                  kernel_initializer='he_normal',
                                  padding='same'))

    # Block 9
    transconv9.append(tf.keras.layers.Conv3DTranspose(16, (2, 2, 2), 
                                          strides=(2, 2, 2), 
                                          padding='same'))
    conv9_1.append(tf.keras.layers.Conv3D(16, (3, 3, 3), 
                                  activation='relu', 
                                  kernel_initializer='he_normal',
                                  padding='same'))
    conv9_2.append(tf.keras.layers.Conv3D(16, (3, 3, 3), 
                                  activation='relu', 
                                  kernel_initializer='he_normal',
                                  padding='same'))

    # Output 
    outconv.append(tf.keras.layers.Conv3D(6, (1, 1, 1), activation='softmax'))

    c1 = conv1_1[i](inputs[i])
    c1 = tf.keras.layers.Dropout(0.2)(c1)
    c1 = conv1_2[i](c1)
    p1 = maxpool1[i](c1)

    # P1  

    c2 = conv2_1[i](p1)
    c2 = tf.keras.layers.Dropout(0.2)(c2)
    c2 = conv2_2[i](c2)
    p2 = maxpool2[i](c2)

    # P2  

    c3 = conv3_1[i](p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = conv3_2[i](c3)
    p3 = maxpool3[i](c3)

    # P3

    c4 = conv4_1[i](p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = conv4_2[i](c4)
    p4 = maxpool4[i](c4)

    # P4

    c5 = conv5_1[i](p4)
    c5 = tf.keras.layers.Dropout(0.2)(c5)
    c5 = conv5_2[i](c5)

    # C5

    u6 = transconv6[i](c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = conv6_1[i](u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = conv6_2[i](c6)

    # C6

    u7 = transconv7[i](c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = conv7_1[i](u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = conv7_2[i](c7)
    

    # C7

    u8 = transconv8[i](c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = conv8_1[i](u8)
    c8 = tf.keras.layers.Dropout(0.2)(c8)
    c8 = conv8_2[i](c8)

    # C8

    u9 = transconv9[i](c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = conv9_1[i](u9)
    c9 = tf.keras.layers.Dropout(0.2)(c9)
    c9 = conv9_2[i](c9)

    # C9  
    outputs = outconv[i](c9)

    nextinp = tf.keras.layers.Conv3D(6, (3, 3, 3), trainable=False, padding='same', 
                                      kernel_initializer='ones', bias_initializer='zeros')(outputs)
    nextinp = tf.keras.layers.concatenate([nextinp, inp])
    inputs.append(nextinp)

  model = tf.keras.Model(inputs=[inp], outputs=[outputs])
  model.compile(**cfg['net_cmp'])
  return model

def dense_unet(iw, ih, ide, ic):
  inputs = tf.keras.layers.Input((iw, ih, ide, ic))

  c1_1 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
  c1_1 = tf.keras.layers.Dropout(0.2)(c1_1)
  c1_1 = tf.keras.layers.concatenate([inputs, c1_1])
  c1_2 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1_1)
  c1_2 = tf.keras.layers.concatenate([c1_1, c1_2])
  p1 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c1_2)

  # P1  

  c2_1 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
  c2_1 = tf.keras.layers.Dropout(0.2)(c2_1)
  c2_1 = tf.keras.layers.concatenate([p1, c2_1])
  c2_2 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2_1)
  c2_2 = tf.keras.layers.concatenate([c2_1, c2_2])
  p2 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c2_2)

  # P2  

  c3_1 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
  c3_1 = tf.keras.layers.Dropout(0.2)(c3_1)
  c3_1 = tf.keras.layers.concatenate([p2, c3_1])
  c3_2 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3_1)
  c3_2 = tf.keras.layers.concatenate([c3_1, c3_2])
  p3 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c3_2)

  # P3

  c4_1 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
  c4_1 = tf.keras.layers.Dropout(0.2)(c4_1)
  c4_1 = tf.keras.layers.concatenate([p3, c4_1])
  c4_2 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4_1)
  c4_2 = tf.keras.layers.concatenate([c4_1, c4_2])
  p4 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c4_2)

  # P4

  c5_1 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
  c5_1 = tf.keras.layers.Dropout(0.2)(c5_1)
  c5_1 = tf.keras.layers.concatenate([p4, c5_1])
  c5_2 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5_1)
  c5_2 = tf.keras.layers.concatenate([c5_1, c5_2])

  # C5

  u6 = tf.keras.layers.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5_2)
  u6 = tf.keras.layers.concatenate([u6, c4_2])
  c6_1 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
  c6_1 = tf.keras.layers.Dropout(0.2)(c6_1)
  c6_1 = tf.keras.layers.concatenate([u6, c6_1])
  c6_2 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6_1)
  c6_2 = tf.keras.layers.concatenate([c6_1, c6_2])

  # C6

  u7 = tf.keras.layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6_2)
  u7 = tf.keras.layers.concatenate([u7, c3_2])
  c7_1 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
  c7_1 = tf.keras.layers.Dropout(0.2)(c7_1)
  c7_1 = tf.keras.layers.concatenate([u7, c7_1])
  c7_2 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7_1)
  c7_2 = tf.keras.layers.concatenate([c7_1, c7_2])

  # C7

  u8 = tf.keras.layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7_2)
  u8 = tf.keras.layers.concatenate([u8, c2_2])
  c8_1 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
  c8_1 = tf.keras.layers.Dropout(0.2)(c8_1)
  c8_1 = tf.keras.layers.concatenate([u8, c8_1])
  c8_2 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8_1)
  c8_2 = tf.keras.layers.concatenate([c8_1, c8_2])

  # C8

  u9 = tf.keras.layers.Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8_2)
  u9 = tf.keras.layers.concatenate([u9, c1_2])
  c9_1 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
  c9_1 = tf.keras.layers.Dropout(0.2)(c9_1)
  c9_1 = tf.keras.layers.concatenate([u9, c9_1])
  c9_2 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9_1)
  c9_2 = tf.keras.layers.concatenate([c9_1, c9_2])

  # C9

  outputs = tf.keras.layers.Conv3D(6, (1, 1, 1), activation='softmax')(c9_2)

  # print(outputs.shape)
  model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
  # model.compile(**cfg['net_cmp'])
  return model

def smooth_filter_init():
  return np.ones([3, 3, 3])

def deep_dense_unet(iw, ih, ide, ic):
  inputs = tf.keras.layers.Input((iw, ih, ide, ic))

  c1_1 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
  c1_1 = tf.keras.layers.Dropout(0.2)(c1_1)
  c1_1 = tf.keras.layers.concatenate([inputs, c1_1])
  c1_2 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1_1)
  c1_2 = tf.keras.layers.concatenate([c1_1, c1_2])
  p1 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c1_2)

  # P1  

  c2_1 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
  c2_1 = tf.keras.layers.Dropout(0.2)(c2_1)
  c2_1 = tf.keras.layers.concatenate([p1, c2_1])
  c2_2 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2_1)
  c2_2 = tf.keras.layers.concatenate([c2_1, c2_2])
  p2 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c2_2)

  # P2  

  c3_1 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
  c3_1 = tf.keras.layers.Dropout(0.2)(c3_1)
  c3_1 = tf.keras.layers.concatenate([p2, c3_1])
  c3_2 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3_1)
  c3_2 = tf.keras.layers.concatenate([c3_1, c3_2])
  p3 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c3_2)

  # P3

  c4_1 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
  c4_1 = tf.keras.layers.Dropout(0.2)(c4_1)
  c4_1 = tf.keras.layers.concatenate([p3, c4_1])
  c4_2 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4_1)
  c4_2 = tf.keras.layers.concatenate([c4_1, c4_2])
  p4 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c4_2)

  # P4

  c5_1 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
  c5_1 = tf.keras.layers.Dropout(0.2)(c5_1)
  c5_1 = tf.keras.layers.concatenate([p4, c5_1])
  c5_2 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5_1)
  c5_2 = tf.keras.layers.concatenate([c5_1, c5_2])

  # C5

  u6 = tf.keras.layers.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5_2)
  u6 = tf.keras.layers.concatenate([u6, c4_2])
  c6_1 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
  c6_1 = tf.keras.layers.Dropout(0.2)(c6_1)
  c6_1 = tf.keras.layers.concatenate([u6, c6_1])
  c6_2 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6_1)
  c6_2 = tf.keras.layers.concatenate([c6_1, c6_2])

  # C6

  u7 = tf.keras.layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6_2)
  u7 = tf.keras.layers.concatenate([u7, c3_2])
  c7_1 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
  c7_1 = tf.keras.layers.Dropout(0.2)(c7_1)
  c7_1 = tf.keras.layers.concatenate([u7, c7_1])
  c7_2 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7_1)
  c7_2 = tf.keras.layers.concatenate([c7_1, c7_2])

  # C7

  u8 = tf.keras.layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7_2)
  u8 = tf.keras.layers.concatenate([u8, c2_2])
  c8_1 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
  c8_1 = tf.keras.layers.Dropout(0.2)(c8_1)
  c8_1 = tf.keras.layers.concatenate([u8, c8_1])
  c8_2 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8_1)
  c8_2 = tf.keras.layers.concatenate([c8_1, c8_2])

  # C8

  u9 = tf.keras.layers.Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8_2)
  u9 = tf.keras.layers.concatenate([u9, c1_2])
  c9_1 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
  c9_1 = tf.keras.layers.Dropout(0.2)(c9_1)
  c9_1 = tf.keras.layers.concatenate([u9, c9_1])
  c9_2 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9_1)
  c9_2 = tf.keras.layers.concatenate([c9_1, c9_2])

  # C9

  outputs = tf.keras.layers.Conv3D(6, (1, 1, 1), activation='softmax')(c9_2)

  nextinp = tf.keras.layers.Conv3D(6, (3, 3, 3), trainable=False, padding='same')(outputs)
  nextinp = tf.keras.layers.concatenate([nextinp, inputs])

  c1_1 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(nextinp)
  c1_1 = tf.keras.layers.Dropout(0.2)(c1_1)
  c1_1 = tf.keras.layers.concatenate([inputs, c1_1])
  c1_2 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1_1)
  c1_2 = tf.keras.layers.concatenate([c1_1, c1_2])
  p1 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c1_2)

  # P1  

  c2_1 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
  c2_1 = tf.keras.layers.Dropout(0.2)(c2_1)
  c2_1 = tf.keras.layers.concatenate([p1, c2_1])
  c2_2 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2_1)
  c2_2 = tf.keras.layers.concatenate([c2_1, c2_2])
  p2 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c2_2)

  # P2  

  c3_1 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
  c3_1 = tf.keras.layers.Dropout(0.2)(c3_1)
  c3_1 = tf.keras.layers.concatenate([p2, c3_1])
  c3_2 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3_1)
  c3_2 = tf.keras.layers.concatenate([c3_1, c3_2])
  p3 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c3_2)

  # P3

  c4_1 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
  c4_1 = tf.keras.layers.Dropout(0.2)(c4_1)
  c4_1 = tf.keras.layers.concatenate([p3, c4_1])
  c4_2 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4_1)
  c4_2 = tf.keras.layers.concatenate([c4_1, c4_2])
  p4 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c4_2)

  # P4

  c5_1 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
  c5_1 = tf.keras.layers.Dropout(0.2)(c5_1)
  c5_1 = tf.keras.layers.concatenate([p4, c5_1])
  c5_2 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5_1)
  c5_2 = tf.keras.layers.concatenate([c5_1, c5_2])

  # C5

  u6 = tf.keras.layers.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5_2)
  u6 = tf.keras.layers.concatenate([u6, c4_2])
  c6_1 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
  c6_1 = tf.keras.layers.Dropout(0.2)(c6_1)
  c6_1 = tf.keras.layers.concatenate([u6, c6_1])
  c6_2 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6_1)
  c6_2 = tf.keras.layers.concatenate([c6_1, c6_2])

  # C6

  u7 = tf.keras.layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6_2)
  u7 = tf.keras.layers.concatenate([u7, c3_2])
  c7_1 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
  c7_1 = tf.keras.layers.Dropout(0.2)(c7_1)
  c7_1 = tf.keras.layers.concatenate([u7, c7_1])
  c7_2 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7_1)
  c7_2 = tf.keras.layers.concatenate([c7_1, c7_2])

  # C7

  u8 = tf.keras.layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7_2)
  u8 = tf.keras.layers.concatenate([u8, c2_2])
  c8_1 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
  c8_1 = tf.keras.layers.Dropout(0.2)(c8_1)
  c8_1 = tf.keras.layers.concatenate([u8, c8_1])
  c8_2 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8_1)
  c8_2 = tf.keras.layers.concatenate([c8_1, c8_2])

  # C8

  u9 = tf.keras.layers.Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8_2)
  u9 = tf.keras.layers.concatenate([u9, c1_2])
  c9_1 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
  c9_1 = tf.keras.layers.Dropout(0.2)(c9_1)
  c9_1 = tf.keras.layers.concatenate([u9, c9_1])
  c9_2 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9_1)
  c9_2 = tf.keras.layers.concatenate([c9_1, c9_2])

  # C9

  outputs = tf.keras.layers.Conv3D(6, (1, 1, 1), activation='softmax')(c9_2)

  # print(outputs.shape)
  model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
  model.compile(**cfg['net_cmp'])
  return model



if __name__ == '__main__':
  model = deep_dense_unet(128, 128, 64, 1)
  print(model.summary())