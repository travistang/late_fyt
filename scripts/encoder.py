import sys
import numpy as np,h5py
from keras.layers import *
from keras.models import Model
from keras.callbacks import *
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from os import listdir

tf.python.control_flow_ops = tf
def get_encoder():
    S = Input(shape = (64,64,3))
    #norm = Lambda(lambda a: a/255.0)(S)

    conv1 = Convolution2D(32,8,8,subsample = (4,4),activation = 'relu',weights = [np.random.rand(3,8,8,32),np.ones((32,)) * 0.01])(S)
    norm1 = BatchNormalization()(conv1)
    conv2 = Convolution2D(64,4,4,subsample = (2,2),activation = 'relu')(norm1)
    norm2 = BatchNormalization()(conv2)
    conv3 = Convolution2D(64,4,4,subsample = (2,2),activation = 'relu')(norm2)
    ############ decoder #########
    up1 = UpSampling2D((4,4))(conv3)
    deconv3 = Convolution2D(64,3,3,activation = 'relu')(up1)
    denorm3 = BatchNormalization()(deconv3)
    up2 = UpSampling2D((3,3))(denorm3)
    deconv2 = Convolution2D(32,4,4,activation = 'relu')(up2)
    denorm2 = BatchNormalization()(deconv2)
    up3 = UpSampling2D((5,5))(deconv2)
    deconv1 = Convolution2D(3,12,12,activation = 'relu')(up3)
    model = Model(input = S,output = deconv1)
    return model
    
model = get_encoder()
model.summary()


imgs = map(lambda im: np.load('../images/' + im),listdir('../images'))
imgs = np.stack(imgs)
rmsprop = RMSprop(lr = 0.001,decay = 1e-7)
model.compile(optimizer=rmsprop,
              loss='mean_squared_error',
              metrics=['accuracy'])
model.fit(imgs,imgs,nb_epoch = 10,validation_split = 0.2)
