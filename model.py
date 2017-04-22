from keras.models import *
from keras.layers import *
from keras.layers.advanced_activations import *
from keras.callbacks import *
from keras.optimizers import Adam
from keras.initializers import *
import tensorflow as tf
from utils import huber_loss

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

def low_guide_v1(lr = 0.0001,num_output = 9):
    S = Input(shape = (116,))
    x = Dense(300,activation = ELU())(S)
    x = Dense(600,activation = ELU())(x)
    x = Dense(num_output,activation = 'linear',init=lambda shape: normal(shape, scale=1e-4))(x)

    model = Model(S,x)
    adam = Adam(lr = lr,decay = 1e-6,clipnorm=0.5)
    model.compile(loss = huber_loss(0.5),optimizer = adam)
    return model

def low_guide_v2(num_action = 1,num_ob = 1):
    # the actor
    S = Input(shape = (1,num_ob))
    x = Flatten()(S)
    x = Dense(300,activation = 'relu')(x)
    x = Dense(600,activation = 'relu')(x)
    x = Dense(num_action,activation = 'linear')(x)

    model = Model(S,x)

    # the critic
    A = Input(shape = (num_action,))
    S = Input(shape = (1,num_ob))
    s = Flatten()(S)
    x = merge([A,s],mode = 'concat')
    x = Dense(300,activation = 'relu')(x)
    x = Dense(600,activation = 'relu')(x)
    x = Dense(1,activation = 'linear')(x)

    critic = Model([A,S],x)
    return model,critic

def img_guide_v1(num_action = 1):
    S = Input(shape = (1,64,64,3))
    x = Reshape((64,64,3))(S)
    x = Conv2D(16,(8,8),strides = (4,4),activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32,(4,4),strides = (2,2),activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32,(4,4),strides = (2,2),activation = 'relu')(x)
    x = Flatten()(x)
    x = Dense(600,activation = 'relu')(x)
    x = Dense(300,activation = 'relu')(x)
    x = Dense(num_action,activation = 'linear')(x)
    actor = Model(S,x)

    S = Input(shape = (1,64,64,3))
    A = Input(shape = (num_action,))
    x = Reshape((64,64,3))(S)
    x = Conv2D(16,(8,8),strides = (4,4),activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32,(4,4),strides = (2,2),activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32,(4,4),strides = (2,2),activation = 'relu')(x)
    x = Flatten()(x)
    x = merge([A,x],mode = 'concat')
    x = Dense(600,activation = 'relu')(x)
    x = Dense(300,activation = 'relu')(x)
    x = Dense(1,activation = 'linear')(x)

    critic = Model([A,S],x)
    return actor,critic

def img_guide_v2(num_action = 1,hist_len = 4):
    S = Input(shape = (1,64,64,3 * hist_len))
    x = Reshape((64,64,3 * hist_len))(S)
    x = Conv2D(32,(8,8),strides = (4,4),activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32,(4,4),strides = (2,2),activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32,(4,4),strides = (2,2),activation = 'relu')(x)
    x = Flatten()(x)
    x = Dense(800,activation = 'relu')(x)
    x = Dense(300,activation = 'relu')(x)
    x = Dense(num_action,activation = 'linear')(x)
    actor = Model(S,x)

    S = Input(shape = (1,64,64,3 * hist_len))
    A = Input(shape = (num_action,))
    x = Reshape((64,64,3 * hist_len))(S)
    x = Conv2D(32,(8,8),strides = (4,4),activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32,(4,4),strides = (2,2),activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32,(4,4),strides = (2,2),activation = 'relu')(x)
    x = Flatten()(x)
    x = merge([A,x],mode = 'concat')
    x = Dense(800,activation = 'relu')(x)
    x = Dense(300,activation = 'relu')(x)
    x = Dense(1,activation = 'linear')(x)

    critic = Model([A,S],x)
    return actor,critic
def img_guide_v3(num_action = 1,hist_len = 4):
    S = Input(shape = (1,hist_len,64,64,3))
    x = Reshape((hist_len,64,64,3))(S)
    x = TimeDistributed(Conv2D(32,(8,8),strides = (4,4),activation = 'relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(32,(4,4),strides = (2,2),activation = 'relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(32,(4,4),strides = (2,2),activation = 'relu'))(x)
    x = Flatten()(x)
    x = Dense(800,activation = 'relu')(x)
    x = Dense(400,activation = 'relu')(x)
    x = Dense(num_action,activation = 'linear')(x)
    actor = Model(S,x)

    S = Input(shape = (1,hist_len,64,64,3))
    A = Input(shape = (num_action,))
    x = Reshape((hist_len,64,64,3))(S)
    x = TimeDistributed(Conv2D(32,(8,8),strides = (4,4),activation = 'relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(32,(4,4),strides = (2,2),activation = 'relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(32,(4,4),strides = (2,2),activation = 'relu'))(x)
    x = Flatten()(x)
    x = merge([A,x],mode = 'concat')
    x = Dense(800,activation = 'relu')(x)
    x = Dense(400,activation = 'relu')(x)
    x = Dense(1,activation = 'linear')(x)

    critic = Model([A,S],x)
    return actor,critic

def stack_model(num_action = 1,hist_len = 4, num_filters = 16):
    S = Input(shape = (1,64,64,3 * hist_len))
    x = Reshape((64,64,3 * hist_len))(S)
    x = Conv2D(num_filters,(8,8),strides = (4,4),activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32,(4,4),strides = (2,2),activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32,(4,4),strides = (2,2),activation = 'relu')(x)
    x = Flatten()(x)
    x = Dense(600 if num_filters == 16 else 800,activation = 'relu')(x)
    x = Dense(300,activation = 'relu')(x)
    x = Dense(num_action,activation = 'linear')(x)
    actor = Model(S,x)

    S = Input(shape = (1,64,64,3 * hist_len))
    A = Input(shape = (num_action,))
    x = Reshape((64,64,3 * hist_len))(S)
    x = Conv2D(num_filters,(8,8),strides = (4,4),activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32,(4,4),strides = (2,2),activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32,(4,4),strides = (2,2),activation = 'relu')(x)
    x = Flatten()(x)
    x = merge([A,x],mode = 'concat')
    x = Dense(600 if num_filters == 16 else 800,activation = 'relu')(x)
    x = Dense(300,activation = 'relu')(x)
    x = Dense(1,activation = 'linear')(x)

    critic = Model([A,S],x)
    return actor,critic

def fork_model(num_action = 1,hist_len = 4, num_filters = 16):
    S = Input(shape = (1,hist_len,64,64,3))
    x = Reshape((hist_len,64,64,3))(S)
    x = TimeDistributed(Conv2D(num_filters,(8,8),strides = (4,4),activation = 'relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(32,(4,4),strides = (2,2),activation = 'relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(32,(4,4),strides = (2,2),activation = 'relu'))(x)
    x = Flatten()(x)
    x = Dense(600 if num_filters == 16 else 800,activation = 'relu')(x)
    x = Dense(400,activation = 'relu')(x)
    x = Dense(num_action,activation = 'linear')(x)
    actor = Model(S,x)

    S = Input(shape = (1,hist_len,64,64,3))
    A = Input(shape = (num_action,))
    x = Reshape((hist_len,64,64,3))(S)
    x = TimeDistributed(Conv2D(num_filters,(8,8),strides = (4,4),activation = 'relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(32,(4,4),strides = (2,2),activation = 'relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(32,(4,4),strides = (2,2),activation = 'relu'))(x)
    x = Flatten()(x)
    x = merge([A,x],mode = 'concat')
    x = Dense(800 if num_filters == 16 else 1200,activation = 'relu')(x)
    x = Dense(400,activation = 'relu')(x)
    x = Dense(1,activation = 'linear')(x)

    critic = Model([A,S],x)
    return actor,critic

def LSTM_model(num_action = 1,hist_len = 4, num_filters = 16):
    S = Input(shape = (1,hist_len,64,64,3))
    x = Reshape((hist_len,64,64,3))(S)
    x = TimeDistributed(Conv2D(num_filters,(8,8),strides = (4,4),activation = 'relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(32,(4,4),strides = (2,2),activation = 'relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(32,(4,4),strides = (2,2),activation = 'relu'))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(100 if num_filters == 16 else 200,activation = 'relu')(x)
    x = Dense(1,activation = 'linear')(x)
    actor = Model(S,x)

    S = Input(shape = (1,hist_len,64,64,3))
    A = Input(shape = (num_action,))
    x = Reshape((hist_len,64,64,3))(S)
    x = TimeDistributed(Conv2D(num_filters,(8,8),strides = (4,4),activation = 'relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(32,(4,4),strides = (2,2),activation = 'relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(32,(4,4),strides = (2,2),activation = 'relu'))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(100 if num_filters == 16 else 200,activation = 'relu')(x)
    x = merge([A,x],mode = 'concat')
    x = Dense(50,activation = 'relu')(x)
    x = Dense(1,activation = 'linear')(x)

    critic = Model([A,S],x)
    return actor,critic