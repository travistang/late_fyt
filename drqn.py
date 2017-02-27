from gym_torcs import TorcsEnv
import numpy as np
import cv2
from keras.layers import *
from keras.models import *
import tensorflow as tf
from ReplayBuffer import ReplayBuffer
from collections import deque

PI= 3.14159265359
tf.python.control_flow_ops = tf
seq_len = 4
def extract_images(vision):
    img = np.ndarray((64,64,3))
    for i in range(3):
        img[:, :, i] = 255 - vision[:, i].reshape((64, 64))
    img = np.flipud(img)
    img = img.transpose(2,0,1)
    return img.astype(np.float32)

def get_network(seq_len = 4):
    S = Input((seq_len,64,64,3))
    conv1 = TimeDistributed(Conv2D(32,8,8,subsample = (4,4),activation = 'relu',weights = [np.random.randn(8,8,3,32),np.ones((32,)) * 0.01]))(S)
    norm1 = TimeDistributed(BatchNormalization())(conv1)
    conv2 = TimeDistributed(Convolution2D(64,4,4,subsample = (2,2),activation = 'relu',weights = [np.random.randn(4,4,32,64),np.ones((64,)) * 0.01]))(norm1)
    norm2 = TimeDistributed(BatchNormalization())(conv2)
    conv3 = TimeDistributed(Convolution2D(64,4,4,subsample = (2,2),activation = 'relu',weights = [np.random.randn(4,4,64,64),np.ones((64,)) * 0.01]))(norm2)
    flat = TimeDistributed(Flatten())(conv3)

    a_fc1 = TimeDistributed(Dense(128,activation = 'relu',weights = [np.random.randn(256,128),np.ones((128,)) * 0.01]))(flat)
    a_lstm = LSTM(8,return_sequences = False)(a_fc1)
    a_out = Activation('softmax')(a_lstm)

    q_fc1 = TimeDistributed(Dense(128,activation = 'relu',weights = [np.random.randn(256,128),np.ones((128,)) * 0.01]))(flat)
    q_lstm = LSTM(1,return_sequences = False)(q_fc1)

    net = Model(S,[a_out,q_lstm])
    return net

net = get_network(seq_len)
net.summary()

env = TorcsEnv(vision = True,throttle = False, gear_change = False)
ob = env.reset()
max_steps = 100000
epsilon = 1
EXPLORE = 100000.
buff = ReplayBuffer(100000)
history = History(seq_len)
for epoch in range(2000):
    for i in range(max_steps):
        # retrieve state first
        s_t = ob[-1]
        # prepare input of the network
        img = np.array(vision).reshape(64,64,3).astype(np.float32)/255.0

        # is this step random?
        if np.random.binomial(1,epsilon) == 1:
            # random!
            act = np.array(np.random.uniform(-1,1))
            s_t1,reward,done,info = env.step(act)
            buff.add(s_t,act,reward,s_t1,done)
        # apply action
        env.step(np.array(action))
        if i % seq_len == 0: # train when it is the multiple of `seq_len`
        # train on buffer
            batches = buff.getSequentialBatch(seq_len)
            # evaluate y
            # TODO: this

    # reduce frequency of exploration after each epoch
    if epsilon > 0:
        epsilon -= 1./EXPLORE


    if np.mod(i, 3) == 0:
        ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
    else:
        ob = env.reset()


'''
    focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel,_ = ob
    # acceleration control
    if speedX < target_speed - (R[u'steer']*50):
        action[1] += 0.1
    else:
        action[1] -= .01
#    if speedX<10:
#       action[1] += 1/(speedX + .1)
    if ((S[u'wheelSpinVel'][2]+S[u'wheelSpinVel'][3]) -
       (S[u'wheelSpinVel'][0]+S[u'wheelSpinVel'][1]) > 5):
       action[1] -= .2
    # check acceleration range
    if  0 > action[1]:
        action[1] = 0
    elif action[1] > 1:
        action[1] = 1
'''
