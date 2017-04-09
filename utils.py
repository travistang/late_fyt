import tensorflow as tf
import numpy as np
from random import choice,sample
# Input: weights of the filter (numpy tensor)
#        tensorflow session
# Under development, DO NOT USE THIS
def conv_weights_summary(weights,sess,name_scope,id):
    # 3 dimensions
    h, w,n_filt = weights.shape
    with tf.name_scope(name_scope):
        filt_ph = tf.placeholder(tf.float32,(h,w,n_filt))
    if n_filt == 3:
        # RGB image
        s = tf.summary.image('conv-%d' % id,filt_ph)
    filts = tf.split(0,n_filt,weights)
    with tf.name_scope(name_scope):
        tf.placeholder(tf.float32,(h,w))

def esp_process(ob,a_t,esp):
    esp = 0. if esp < 0 else 1. if esp > 1. else esp # clamp epsilon
    ran = np.random.binomial(2,esp)
    should_not_guide = (ran == 0)
    should_random = (ran == 2)
    return (a_t + np.random.normal(0,0.4),True) if should_random else (a_t,False) if should_not_guide else (guided_action(ob),None) # None for guided

def discrete_esp_process(esp,org_input,num_output):
    esp = 0. if esp < 0 else 1. if esp > 1. else esp # clamp epsilon
    ran = np.random.binomial(1,esp)
    return org_input if ran == 0 else sample(range(num_output))

def get_esp(step,cur_esp):
    return 1. - step * 1./10000

def get_low_states(ob):
    return np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
def get_states(ob,history,img_states = True):
    states = np.concatenate(history,axis = -1)
    if img_states:
        img = get_image(ob)
        history.append(img)
    else:
        l_s = get_low_states(ob)
        history.append(l_s)    
    history.popleft()
    return states

# given an output of a "discreted" actor, return the continuous steering command
def get_inferred_steering(pred_out): 
    pred_out = pred_out.squeeze() # reduce the batch dimension
    num_states = pred_out.shape[0]
    con_steerings = np.arange(-1,1,num_states)
    max_q = np.argmax(pred_out)
    return np.linspace(-1,1,10)

# given a batch sampled from the replay buffer, return discounted reward and states for backpropagation
def preprocess_batch(batch,actor,gamma,batch_size):
    # info extraction
    st = np.array(map(lambda e: e[0],batch))
    a_t = np.array(map(lambda e: e[1],batch))
    st1 = np.array(map(lambda e: e[3],batch))
    dones = np.array(map(lambda e: e[4],batch))
    r = np.array(map(lambda e: e[2],batch))

    q_t1 = actor.predict(st1).squeeze()
    y = [r[i] + gamma * q_t1[i] if not dones[i] else r[i] for i in range(batch_size)]
    return st,y

################################################
# Tensorflow summary helper
class SummaryManager(object):
    def __init__(self,dir = 'tmp/discrete'):
        self.summaries = []
        self.placeholders = []
        self.writer = self.summary.FileWriter(dir)
        self.sess = tf.Session()

    def add_scalar_summary(self,name,ph_shape = None,name_scope = None):
        if name_scope:
            with tf.name_scope(name_scope):
                ph = tf.placeholder(tf.float32,ph_shape,name = name)
        else:
            ph = tf.placeholder(tf.float32,ph_shape,name = name)
        self.placeholders.append(ph)
        self.summaries.append(tf.summary.scalar(name,self.placeholders[-1]))

    def write_summary(self,feed_dict):
        pair = dict()
        for name,val in feed_dict:
            pair[name + ':0'] = val # since feed_dict recognizes name with
        sum_op = tf.summary.merge(self.summaries)
        r = self.sess.run(sum_op,feed_dict = pair)
        self.writer.write(r)