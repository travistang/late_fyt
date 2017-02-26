from keras.models import Model
from keras.layers import *
import keras.backend as K
import tensorflow as tf
tf.python.control_flow_ops = tf
S = Input(shape=(3,64,64))
conv1 = Convolution2D(16,3,3)(S)
lrn1 = BatchNormalization()(conv1)
act1 = Activation('relu')(lrn1)
pool1 = MaxPooling2D(pool_size = (3,3))(act1)

conv2 = Convolution2D(32,3,3)(pool1)
lrn2 = BatchNormalization()(conv2)
act2 = Activation('relu')(lrn2)

conv3 = Convolution2D(64,3,3)(act2)
lrn3 = BatchNormalization()(conv3)
act3 = Activation('relu')(lrn3)
      
flat = Flatten()(act3)

fc1 = Dense(1000,init = 'normal')(flat)

net = Model(input = S, output = fc1)
net.summary()
