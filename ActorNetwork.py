import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.engine.training import collect_trainable_weights
from keras.regularizers import *
from keras.layers import *
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=(64,64,4))
        conv1 = Convolution2D(32,8,8,subsample = (4,4),activation = 'relu',weights = [np.random.uniform(-1./64,1./64,size = (8,8,4,32)),np.random.uniform(1./64,2./64,size = (32,))])(S)
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
        Steering = Dense(1,weights = [np.random.uniform(0,0,(512,1)),np.zeros((1,))], name='Steering')(lrn4)
        model = Model(S,Steering)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, model.trainable_weights, S
