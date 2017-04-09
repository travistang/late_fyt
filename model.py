from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import Adam
import tensorflow as tf


def guide_v1():
        S = Input(shape = (64,64,12))
        x = Convolution2D(32,8,8,subsample = (4,4),activation = 'relu')(S)
        x = BatchNormalization()(x)
        x = Convolution2D(32,4,4,subsample = (2,2),activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = Convolution2D(64,4,4,subsample = (2,2),activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
#        z = Dense(128,init = 'uniform',activation = 'relu',name = 'ls_1',trainable = False)(x)
#        ls = Dense(29,init = 'uniform',activation = 'relu',name = 'ls_2',trainable = False)(z)

        y = Dense(300,activation = 'relu',name = 'act_1')(x)
        Steering = Dense(1,activation = 'linear',name = 'act_2')(y)
        #Steering = Dense(1,weights = [np.random.uniform(-1e-8,1e-8,(512,1)),np.zeros((1,))], name='Steering')(lrn4)
        model = Model(S,Steering)
        adam = Adam(lr=0.00000001,decay = 1e-6)
        K.get_session().run([adam.beta_1.initializer,adam.beta_2.initializer])
        model.compile(loss='mse', optimizer=adam)
        if weight_files:
            model.load_weights(weight_files)
        return model, model.trainable_weights, S


def guide_v2():
    S = Input(shape = (64,64,4))
    x = Convolution2D(32,8,8,subsample = (4,4),activation = 'relu')(S)
    x = BatchNormalization()(x)
    x = Convolution2D(32,4,4,subsample = (2,2),activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(32,4,4,subsample = (2,2),activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(300,activation = 'relu')(x)
    x = Dense(8,activation = 'linear',name = 'act_2')(x)
    model = Model(S,x)
    adam = Adam(lr = 0.0001,decay = 1e-6)
    model.compile(loss = 'categorial_accuracy',optimizer = adam)
    return model

def low_guide_v1(lr = 0.0001):
    S = Input(shape = (116,))
    x = Dense(50,activation = 'relu')(S)
    x = Dense(10,activation = 'linear')(x)

    model = Model(S,x)
    adam = Adam(lr = lr,decay = 1e-6)
    model.compile(loss = 'mse',optimizer = adam)
    return model