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

def get_image(ob):
    img = ob[-1]
    vision = ob[-1]
    img = np.array(vision).reshape(64,64,3)
    img = np.flipud(img).astype(np.float32)/255.0 # normalized, colored image
    #img = cv2.cvtColor(np.flipud(img).astype(np.float32),cv2.COLOR_BGR2GRAY)/255.0
    return img

def discrete_esp_process(esp,org_input,num_output):
    esp = 0. if esp < 0 else 1. if esp > 1. else esp # clamp epsilon
    ran = np.random.binomial(1,esp)
    spaces = np.linspace(-1,1,num_output)
    ind = np.random.binomial(num_output - 1,0.5)
    return (org_input,True) if ran == 0 else (spaces[ind],False) # the boolean indicates whether a random process is chosen

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
def get_inferred_steering(pred_out,num_output): 
    con_steerings = np.linspace(-1,1,num_output)
    max_q = np.argmax(pred_out,axis = 1)[0]
    print max_q
    return con_steerings[max_q]

def get_indices_of_continuous_output(con_output,num_output):
    spaces = np.linspace(-1,1,num_output)
    for i,con_out in enumerate(spaces):
        if con_out == con_output:
            return i
    assert False # cannot let this go

def update_network(net,target_net,tau):
    w = net.get_weights()
    tw = target_net.get_weights()
    for i in range(len(w)):
        tw[i] = tau * w[i] + tw[i] * (1 - tau)
    target_net.set_weights(tw)

# given a batch sampled from the replay buffer, return discounted reward and states for backpropagation
def preprocess_batch(batch,actor,gamma,batch_size,target_actor = None):
    # info extraction
    st = np.array(map(lambda e: e[0],batch))
    a_t = map(lambda e: e[1],batch)
    st1 = np.array(map(lambda e: e[3],batch))
    dones = map(lambda e: e[4],batch)
    r = np.array(map(lambda e: e[2],batch))

    q_t1 = actor.predict(st1)
    actor_act = np.argmax(q_t1,axis = 1)
    q_t1 = np.max(q_t1,axis = 1)

    #for k,q in enumerate(q_t1): print 'q%d:%s' % (k,str(q.shape))
    if target_actor is None:
        y = [r[i] + gamma * q_t1[i] if not dones[i] else r[i] for i in range(batch_size)]
    else:
        # Q values of each actions estimated by the target actor in st1 
        tar_q = actor.predict(st1)
        # get the estimated Q value of the action taken by the actor from the target actor 
        y = [r[i] + gamma * tar_q[i][actor_act[i]] if not dones[i] else r[i] for i in range(batch_size)]
    return st,y,a_t

def transfer_weights(from_net,to_net):
    w = from_net.get_weights()
    to_net.set_weights(w)

# remain the output of the network,except the max. prediction of the actor
def prepare_label(st,a_t,y,actor,num_output):
    Q = actor.predict(st)
    inds = map(lambda a:get_indices_of_continuous_output(a,num_output),a_t)
    
    for row,ind in enumerate(inds):
        Q[row][ind] = y[row]
    return Q

################################################
# Tensorflow summary helper
class SummaryManager(object):
    def __init__(self,dir = 'tmp/discrete'):
        self.summaries = []
        self.placeholders = []
        self.writer = tf.summary.FileWriter(dir)
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
        for name,val in feed_dict.iteritems():
            pair[name + ':0'] = val # since feed_dict recognizes name with
        sum_op = tf.summary.merge(self.summaries)
        r = self.sess.run(sum_op,feed_dict = pair)
        self.writer.write(r)

def huber_loss(clip_value):
    return lambda y_pred,y_true: tf.cond(tf.abs(tf.reduce_sum(y_pred - y_true)) < clip_value, 
        lambda: 0.5 * tf.square(tf.reduce_sum(y_pred - y_true)),
        lambda: clip_value * (tf.abs(tf.reduce_sum(y_pred - y_true)) - 0.5 * clip_value))
