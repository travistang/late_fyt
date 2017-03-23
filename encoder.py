from keras.models import Model
from keras.layers import *
import keras.backend as K
import tensorflow as tf
from keras.optimizers import *
from keras import backend as K
from keras.models import *
from keras.callbacks import *
tf.python.control_flow_ops = tf
def get_encoder():
    S = Input(shape=(64,64,1))
    conv1 = Convolution2D(32,8,8,subsample = (4,4),activation = 'relu',weights = [np.random.uniform(-1./256,1./256,size = (8,8,1,32)),np.random.uniform(1./256,2./256,size = (32,))])(S)
    lrn1 = BatchNormalization()(conv1)

    conv2 = Convolution2D(64,4,4,subsample = (2,2),activation = 'relu',weights = [np.random.uniform(-1./np.sqrt(32 * 8 * 8),1./np.sqrt(32 * 8 * 8),size = (4,4,32,64)),np.random.uniform(1./np.sqrt(32 * 8 * 8),2./np.sqrt(32 * 8 * 8)\
            ,size = (64,))])(lrn1)
    lrn2 = BatchNormalization()(conv2)

    conv3 = Convolution2D(64,3,3,subsample = (1,1),weights = [np.random.uniform(-1./np.sqrt(64 * 4 * 4),1./np.sqrt(64 * 4 * 4),(3,3,64,64)),np.random.uniform(1./np.sqrt(64 * 4 * 4)\
        ,2./np.sqrt(64 * 4 * 4),(64,))],activation = 'relu')(lrn2)
    lrn3 = BatchNormalization()(conv3)

    flat = Flatten()(lrn3)
    drop = Dense(512,activation = 'relu',weights = [np.random.uniform(-1e-3,1e-3,(1024,512)),np.random.uniform(1e-3,2e-3,(512,))])(flat)
    lrn4 = BatchNormalization()(drop)
    re1 = Reshape((8,8,8))(lrn4)
    deconv3 = Convolution2D(64,3,3,subsample = (1,1),activation = 'relu')(re1)
    up2 = UpSampling2D((3,3))(deconv3)
    de_lrn2 = BatchNormalization()(up2)
    deconv2 = Convolution2D(32,4,4,subsample = (1,1),activation = 'relu')(de_lrn2)
    up1 = UpSampling2D((5,5))(deconv2)
    de_lrn1 = BatchNormalization()(up1)
    rep = Convolution2D(1,12,12,activation = 'relu')(de_lrn1)
    net = Model(S,rep)
    return net

def train_encoder(net):
    from os import listdir
    import numpy as np
    imgs = map(lambda f: np.load(f).reshape(64,64,1),filter(lambda f: '.npy' in f,listdir('.')))
    X = np.stack(imgs)

    # compile net
    rmsprop = RMSprop(lr = 0.0005,decay = 1e-6)
    net.compile(loss = 'mean_squared_error',optimizer = rmsprop)
    K.get_session().run(tf.global_variables_initializer())
    net.fit(X,X,nb_epoch = 50,batch_size = 16,validation_split = 0.2,callbacks = [EarlyStopping(patience = 1)])
    net.save('encoder.h5')

'''
def get_pretrained_encoder():
    trained = load_model('encoder.h5')
    S = Input(shape=(4,64,64,3))
    conv1 = TimeDistributed(Convolution2D(16,8,8,subsample = (4,4),init = 'uniform',activation = 'relu'))(S)
    lrn1 = TimeDistributed(BatchNormalization())(conv1)

    conv2 = TimeDistributed(Convolution2D(32,4,4,subsample = (2,2),init = 'uniform',activation = 'relu'))(lrn1)
    lrn2 = TimeDistributed(BatchNormalization())(conv2)

    conv3 = TimeDistributed(Convolution2D(32,3,3,subsample = (1,1),init = 'uniform',activation = 'relu'))(lrn2)
    lrn3 = TimeDistributed(BatchNormalization())(conv3)

    flat = TimeDistributed(Flatten())(lrn3)
    drop = TimeDistributed(Dense(128,init = 'uniform',activation = 'relu'))(flat)
    lrn4 = TimeDistributed(BatchNormalization())(drop)

    model = Model(S,lrn4)
    #for i,l in enumerate(model.layers):
    #    l.set_weights(trained.layers[i].get_weights())
    #   l.trainable = False # freeze train layers
    return model,S,lrn4
'''

def get_pretrained_encoder():
    S = Input(shape = (4,64,64,3))
    x = TimeDistributed(Convolution2D(32,5,5,subsample = (1,1),init = 'uniform',activation = 'relu'))(S)
    x = TimeDistributed(BatchNormalization())(x)


    x = TimeDistributed(Convolution2D(64,5,5,subsample = (1,1),init = 'uniform',activation = 'relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)



    x = TimeDistributed(Convolution2D(64,5,5,subsample = (3,3),init = 'uniform',activation = 'relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(Convolution2D(256,3,3,subsample = (3,3),init = 'uniform',activation = 'relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(Convolution2D(256,1,1,subsample = (1,1),init = 'uniform',activation = 'relu'))(x)
    x = TimeDistributed(Convolution2D(128,1,1,subsample = (1,1),init = 'uniform',activation = 'relu'))(x)
    x = TimeDistributed(Convolution2D(64,1,1,subsample = (1,1),init = 'uniform',activation = 'relu'))(x)
    x = TimeDistributed(Convolution2D(32,1,1,subsample = (1,1),init = 'uniform',activation = 'relu'))(x)
    x = TimeDistributed(Convolution2D(16,1,1,subsample = (1,1),init = 'uniform',activation = 'relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Flatten())(x)
    model = Model(S,x)
    model.summary()
    return model,S,x

if __name__ == '__main__':
    trained = load_model('encoder.h5')
    model = get_pretrained_encoder()
    import numpy as np
    import theano as T
    sess = tf.InteractiveSession()
    #K._LEARNING_PHASE = tf.Constant(0)

    X = np.random.rand(1,64,64,1)
    res = model.predict(X)
    t_res = trained.predict(X)
#    print model.layers[1].get_weights()[0],trained.layers[1].get_weights()[0]
#    print 'match' if np.array_equal(model.layers[1].get_weights()[0],trained.layers[1].get_weights()[0]) else 'gg'
