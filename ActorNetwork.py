import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.engine.training import collect_trainable_weights
from keras.regularizers import *
from keras.layers import *
from keras.optimizers import Adam,RMSprop
import tensorflow as tf
import keras.backend as K
from encoder import get_pretrained_encoder
from keras.callbacks import *
HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE,log_path = 'tmp/actor'):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.summaries = []
        with tf.name_scope("action_gradient"):
            self.action_gradient = tf.placeholder(tf.float32,[None, action_size]) # gradient of critic...
            self.summaries.append(tf.summary.scalar('action_gradient',tf.norm(self.action_gradient)))
        self.params_grad = tf.gradients(self.model.output[0], self.weights, -self.action_gradient) # grad_u ()
        #grads = zip([tf.clip_by_norm(grad,5) for grad in self.params_grad], self.weights)
        grads = zip(self.params_grad,self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        '''
        with tf.name_scope('actor_summary'):
            for ind,(clip_grad, w) in enumerate(grads):
                self.summaries.append(tf.summary.scalar('clipped_gradient_norm_%d' % ind,tf.norm(clip_grad)))
        '''
        self.summary_op = tf.summary.merge(self.summaries)
        self.sess.run(tf.initialize_all_variables())

        self.last_action_grads = None # for logging purpose
    def log(self,states,action_grads):
        summary = self.sess.run(self.summary_op,feed_dict = {
            self.state: states,
            self.action_gradient: action_grads
        })
        return summary

    def get_norm(self,states,action_grads):
        grad_norm = [tf.norm(g) for g in self.params_grad]
        return self.sess.run(grad_norm,feed_dict = {
            self.state: states,
            self.action_gradient: action_grads
        })

    def train(self, states, action_grads):
        self.last_action_grads = action_grads
        return self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })
    def get_weight_norm(self):
        layer_with_weights = filter(lambda l: len(l.get_weights()) == 2,self.model.layers)
        weights = map(lambda l: l.get_weights()[0],layer_with_weights)
        return map(lambda w: np.linalg.norm(w),weights)
    def get_weight_varience(self):
        layer_with_weights = filter(lambda l: len(l.get_weights()) == 2,self.model.layers)
        weights = map(lambda l: l.get_weights()[0],layer_with_weights)
        return
    def get_bias_norm(self):
        layer_with_weights = filter(lambda l: len(l.get_weights()) == 2,self.model.layers)
        weights = map(lambda l: l.get_weights()[1],layer_with_weights)
        return map(lambda w: np.linalg.norm(w),weights)
    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        print("Now we build the model")
        from keras.models import load_model
        model = load_model('guide_model.h5')
        for l in model.layers:
            l.trainable = False
        return model,model.trainable_weights,model.input
