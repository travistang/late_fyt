import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json, load_model
from keras.engine.training import collect_trainable_weights

from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=(64,64,1))
        A = Input(shape=(1,),name='action2')
        a_fc1 = Dense(512,activation = 'relu',weights = [np.random.uniform(-1e-4,1e-4,(1,512)),np.zeros((512,))])(A)

        conv1 = Convolution2D(32,8,8,subsample = (4,4),activation = 'relu',weights = [np.random.uniform(-1./64,1./64,size = (8,8,1,32)),np.random.uniform(1./64,2./64,size = (32,))])(S)
        lrn1 = BatchNormalization()(conv1)

        conv2 = Convolution2D(64,4,4,subsample = (2,2),activation = 'relu',weights = [np.random.uniform(-1./np.sqrt(32 * 8 * 8),1./np.sqrt(32 * 8 * 8),size = (4,4,32,64)),np.random.uniform(1./np.sqrt(32 * 8 * 8),2./np.sqrt(32 * 8 * 8)\
            ,size = (64,))])(lrn1)
        lrn2 = BatchNormalization()(conv2)

        conv3 = Convolution2D(64,3,3,subsample = (1,1),weights = [np.random.uniform(-1./np.sqrt(64 * 4 * 4),1./np.sqrt(64 * 4 * 4),(3,3,64,64)),np.random.uniform(1./np.sqrt(64 * 4 * 4)\
            ,2./np.sqrt(64 * 4 * 4),(64,))],activation = 'relu')(lrn2)
        lrn3 = BatchNormalization()(conv3)

        flat = Flatten()(lrn3)
        drop = Dense(512,activation = 'relu',weights = [np.random.uniform(-1e-3,1e-3,(1024,512)),np.random.uniform(1e-3,2e-3,(512,))])(flat)
        drop_norm = BatchNormalization()(drop)
        V = Merge(mode = 'concat')([drop_norm,a_fc1])
        fc_1 = Dense(512,activation = 'relu',weights = [np.random.uniform(-1e-4,1e-4,(1024,512)),np.zeros((512,))])(V)
        fc_1_norm = BatchNormalization()(fc_1)
        Q = Dense(1,weights = [np.random.uniform(-1e-4,1e-4,(512,1)),np.zeros((1,))])(fc_1_norm)
        model = Model(input=[S,A],output=Q)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S
