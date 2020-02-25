import tensorflow as tf
from tensorflow.keras.layers import Add, Multiply, Dense, Input, Conv3D, Dropout, concatenate, MaxPooling3D, Conv3DTranspose, UpSampling3D, GlobalAveragePooling3D
import tensorflow.keras.backend as K

from config import get_config

cfg = get_config()

def conv_block(inp, channels, output_name, block_name='Block', dropout=0.2, depth=2, kernel_size=(3, 3, 3), activation='relu', sSE=False, cSE=False, scSE=False):
    with K.name_scope(block_name):
        c_1 = Conv3D(channels[0], kernel_size, activation='relu', kernel_initializer='glorot_uniform', padding='same')(inp)
        c_1 = Dropout(dropout)(c_1)
        c_1 = concatenate([inp, c_1])
        if scSE or (sSE and cSE):
            c_2 = Conv3D(channels[1], kernel_size, activation=activation, kernel_initializer='glorot_uniform', padding='same')(c_1)
            
            # cSE
            cse = GlobalAveragePooling3D()(c_2)
            cse = Dense(c_2.shape[-1] // 2, activation='relu')(cse)
            cse = Dense(c_2.shape[-1], activation='sigmoid')(cse)
            c_2_cse = c_2 * cse
            
            # sSE
            sse = Conv3D(1, (1, 1, 1), activation='sigmoid', kernel_initializer='glorot_uniform')(c_2)
            c_2_sse = c_2 * sse
            return Add(name=output_name)([c_2_cse, c_2_sse])
        elif cSE:
            sse = Conv3D(1, (1, 1, 1), activation='sigmoid', kernel_initializer='glorot_uniform')(c_2)
            return Multiply([c_2, sse], name=output_name)
        elif sSE:
            # cSE
            cse = GlobalAveragePooling3D()(c_2)
            cse = Dense(c_2.shape[-1] // 2, activation='relu')(cse)
            cse = Dense(c_2.shape[-1], activation='sigmoid')(cse)
            return Multiply([c_2, cse], name=output_name)
        else:
            c_2 = Conv3D(channels[1], kernel_size, activation=activation, kernel_initializer='glorot_uniform', padding='same', name=output_name)(c_1)
            return c_2
            
def att_block(x, g, channels, block_name='AttentionBlock'):
    with K.name_scope(block_name):
        x1 = Conv3D(channels, (1, 1, 1), kernel_initializer='he_uniform', padding='same')(x)
        g1 = Conv3D(channels, (1, 1, 1), kernel_initializer='he_uniform', padding='same')(g)
        psi = tf.keras.layers.Activation('relu')(x1+g1)
        psi = Conv3D(channels, (1, 1, 1), activation='sigmoid', kernel_initializer='he_uniform', padding='same')(psi)
        out = x * psi
    return out
        
def attunet(w, h, d, c, r=1):
    inputs = Input((w, h, d, c))

    inp = inputs

    for i in range(r):

        # Down conv
        
        c1 = conv_block(inp, [16, 16], 'c1')
        p1 = MaxPooling3D((2, 2, 2))(c1)

        # P1

        c2 = conv_block(p1, [32, 32], 'c2')
        p2 = MaxPooling3D((2, 2, 2))(c2)
        
        # P2
        
        c3 = conv_block(p2, [64, 64], 'c3')
        p3 = MaxPooling3D((2, 2, 2))(c3)
        
        # P3

        c4 = conv_block(p3, [128, 128], 'c4')
        p4 = MaxPooling3D((2, 2, 2))(c4)
        
        # P4

        c5 = conv_block(p4, [256, 256], 'c5')
        
        # C5

        u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2,2,2), padding='same')(c5)
        u6 = att_block(u6, c4, 128)
        c6 = conv_block(u6, [128, 128], 'c6')

        # C6

        u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2,2,2), padding='same')(c6)
        u7 = att_block(u7, c3, 64)
        c7 = conv_block(u7, [64, 64], 'c7')

        # C6

        u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2,2,2), padding='same')(c7)
        u8 = att_block(u8, c2, 32)
        c8 = conv_block(u8, [32, 32], 'c8')

        # C6

        u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2,2,2), padding='same')(c8)
        u9 = att_block(u9, c1, 16)
        c9 = conv_block(u9, [16, 16], 'c9')

        outputs = tf.keras.layers.Conv3D(6, (1, 1, 1), activation='softmax')(c9)

        nextinp = tf.keras.layers.Conv3D(6, (3, 3, 3), trainable=False, padding='same')(outputs)
        inp = tf.keras.layers.concatenate([nextinp, inputs])


    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(**cfg['net_cmp'])

    return model

def unet(w, h, d, c, r=1, scSE=False, sSE=False, cSE=False):
    inputs = Input((w, h, d, c))

    inp = inputs
    outputs = []
    for i in range(r):

        # Down conv
        
        c1 = conv_block(inp, [16, 16], 'c1_' + str(i), scSE=scSE, sSE=sSE, cSE=cSE)
        p1 = MaxPooling3D((2, 2, 2))(c1)

        # P1

        c2 = conv_block(p1, [32, 32], 'c2_' + str(i), scSE=scSE, sSE=sSE, cSE=cSE)
        p2 = MaxPooling3D((2, 2, 2))(c2)
        
        # P2
        
        c3 = conv_block(p2, [64, 64], 'c3_' + str(i), scSE=scSE, sSE=sSE, cSE=cSE)
        p3 = MaxPooling3D((2, 2, 2))(c3)
        
        # P3

        c4 = conv_block(p3, [128, 128], 'c4_' + str(i), scSE=scSE, sSE=sSE, cSE=cSE)
        p4 = MaxPooling3D((2, 2, 2))(c4)
        
        # P4

        c5 = conv_block(p4, [256, 256], 'c5_' + str(i), scSE=scSE, sSE=sSE, cSE=cSE)
        
        # C5

        u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2,2,2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = conv_block(u6, [128, 128], 'c6_' + str(i), scSE=scSE, sSE=sSE, cSE=cSE)

        # C6

        u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2,2,2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = conv_block(u7, [64, 64], 'c7_' + str(i), scSE=scSE, sSE=sSE, cSE=cSE)

        # C6

        u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2,2,2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = conv_block(u8, [32, 32], 'c8_' + str(i), scSE=scSE, sSE=sSE, cSE=cSE)

        # C6

        u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2,2,2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        c9 = conv_block(u9, [16, 16], 'c9_' + str(i), scSE=scSE, sSE=sSE, cSE=cSE)

        outputs.append(tf.keras.layers.Conv3D(6, (1, 1, 1), activation='softmax', name='output_{}'.format(i))(c9))

        nextinp = tf.keras.layers.Conv3D(6, (3, 3, 3), trainable=False, padding='same')(outputs[i])
        inp = tf.keras.layers.concatenate([nextinp, inputs])

    losses = {'output_{}'.format(v):'categorical_crossentropy' for v in range(r)}
    loss_weights = {'output_{}'.format(v):1.0 for v in range(r)}

    model = tf.keras.Model(inputs=[inputs], outputs=outputs)
    model.compile(**cfg['net_cmp'], loss=losses, loss_weights=loss_weights)

    return model

def scSEunet(w, h, d, c, r=1):
    return unet(w, h, d, c, r=r, scSE=True)

def cSEunet(w, h, d, c, r=1):
    return unet(w, h, d, c, r=r, cSE=True)

def sSEunet(w, h, d, c, r=1):
    return unet(w, h, d, c, r=r, sSE=True)

def unetpp(w, h, d, c, loss='categorical_crossentropy', scSE=False):
    inputs = Input((w, h, d, c))

    # inp = inputs

    # for i in range(r):

    # Down conv
    with tf.name_scope('X_00'):
        c11 = conv_block(inputs, [16, 16], 'c11', scSE=scSE)
        p11 = MaxPooling3D((2, 2, 2))(c11)

    # P11
    with tf.name_scope('X_10'):
        c21 = conv_block(p11, [32, 32], 'c21', scSE=scSE)
        p21 = MaxPooling3D((2, 2, 2))(c21)

    # P21
    with tf.name_scope('X_01'):
        c12 = UpSampling3D(size=2)(c21)
        c12 = concatenate([c12, c11])
        c12 = conv_block(c12, [16, 6], 'c12', activation='softmax') #c1, scSE=scSE2
    
    # C12
    with tf.name_scope('X_20'):
        c31 = conv_block(p21, [64, 64], 'c31', scSE=scSE)
        p31 = MaxPooling3D((2, 2, 2))(c31)
    
    # P31
    with tf.name_scope('X_11'):
        c22 = UpSampling3D(size=2)(c31)
        c22 = concatenate([c22, c21])
        c22 = conv_block(c22, [32, 32], 'c22', scSE=scSE)
    
    # C22
    with tf.name_scope('X_02'):
        c13 = UpSampling3D(size=2)(c22)
        c13 = concatenate([c13, c12, c11])
        c13 = conv_block(c13, [16, 6], 'c13', activation='softmax', scSE=scSE)
    
    # C13
    with tf.name_scope('X_30'):
        c41 = conv_block(p31, [128, 128], 'c41', scSE=scSE)
        p41 = MaxPooling3D((2, 2, 2))(c41)
    
    # P4
    with tf.name_scope('X_21'):
        c32 = UpSampling3D(size=2)(c41)
        c32 = concatenate([c32, c31])
        c32 = conv_block(c32, [64, 64], 'c32', scSE=scSE)
    
    # C32
    with tf.name_scope('X_12'):
        c23 = UpSampling3D(size=2)(c32)
        c23 = concatenate([c23, c22, c21])
        c23 = conv_block(c23, [32, 32], 'c23', scSE=scSE)
    
    # C23
    with tf.name_scope('X_03'):
        c14 = UpSampling3D(size=2)(c23)
        c14 = concatenate([c14, c13, c12, c11])
        c14 = conv_block(c14, [16, 6], 'c14', activation='softmax', scSE=scSE)
    
    # C23
    with tf.name_scope('X_40'):
        c5 = conv_block(p41, [256, 256], 'c5', scSE=scSE)
    
    # C5
    with tf.name_scope('X_31'):
        u6 = UpSampling3D(size=2)(c5)
        u6 = concatenate([u6, c41])
        c6 = conv_block(u6, [128, 128], 'c6', scSE=scSE)

    # C6
    with tf.name_scope('X_22'):
        u7 = UpSampling3D(size=2)(c6)
        u7 = concatenate([u7, c31, c32])
        c7 = conv_block(u7, [64, 64], 'c7', scSE=scSE)

    # C7
    with tf.name_scope('X_13'):
        u8 = UpSampling3D(size=2)(c7)
        u8 = concatenate([u8, c21, c22, c23])
        c8 = conv_block(u8, [32, 32], 'c8', scSE=scSE)

    # C8
    with tf.name_scope('X_04'):
        u9 = UpSampling3D(size=2)(c8)
        u9 = concatenate([u9, c11, c12, c13, c14])
        # print(u9.shape)
        outputs = conv_block(u9, [16, 6], 'outputs', activation='softmax', scSE=scSE)

    # outputs = tf.keras.layers.Conv3D(6, (1, 1, 1), activation='softmax', name='outputs')(c9)
        # nextinp = tf.keras.layers.Conv3D(6, (3, 3, 3), trainable=False, padding='same')(outputs)
        # inp = tf.keras.layers.concatenate([nextinp, inputs])

    
    model = tf.keras.Model(inputs=[inputs], outputs=[c12, c13, c14, outputs])
    losses = {'{}'.format(o):loss for o in ['c12', 'c13', 'c14', 'outputs']}
    loss_weights = {'{}'.format(o):1 for o in ['c12', 'c13', 'c14', 'outputs']}
    model.compile(**cfg['net_cmp'], loss=losses, loss_weights=loss_weights)

    return model

if __name__ == '__main__':
    pass