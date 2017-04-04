from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import Adam
import tensorflow as tf
from encoder import get_pretrained_encoder
from ddpg import get_image
import keras.backend as K
tf.python.control_flow_ops = tf
K.set_learning_phase(0)
def get_critic():

        HIDDEN1_UNITS = 300
        HIDDEN2_UNITS = 600
        S = Input(shape=[29])
        A = Input(shape=[1],name='action2')
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A)
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = merge([h1,a1],mode='concat')
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(1,activation='linear')(h3)
        model = Model(input=[S,A],output=V)
        adam = Adam(lr = 0.000001,decay=1e-6)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S

def get_action_trainable_weights(guide,isValue = False):
    if not isValue:
        return guide.get_layer('act_1').trainable_weights + guide.get_layer('act_2').trainable_weights
    else:
        return guide.get_layer('act_1').get_weights() + guide.get_layer('act_2').get_weights()
'''
    Util function for applying DDPG to the guide (actor)
    Input:
        guide: the Guide model
        sess: the Tensorflow session
        optimizer: the Tensorflow Optimizer (e.g. AdamOptimizer)
        act_grad: gradient from the critic
'''
def apply_ddpg_gradients(guide,critic_out,critic_action_ph,critic_state_ph,sess,optimizer,action,l_state):
    weights = get_action_trainable_weights(guide)
    action_gradient = tf.gradients(critic_out,critic_action_ph)
    action_gradient = sess.run(action_gradient,feed_dict = {critic_action_ph: action,critic_state_ph: l_state})
    # param_grad_op
    param_grad = tf.gradients(guide.output,weights,[map(lambda a: -a,action_gradient) for i in range(2)]) # some hack here. The action gradient is extended to number of trainable weights
    grads = zip(params_grad,weights)
    sess.run(optimizer.apply_gradients(grad),feed_dict = {
        critic_action_ph: action
        })

def add_noise(action):
    return action + np.random.randn(action.shape)
def get_guide():
        S = Input(shape = (64,64,12))
        x = Convolution2D(64,5,5,subsample = (3,3),init = 'uniform',activation = 'relu')(S)
        x = BatchNormalization()(x)
        x = Convolution2D(64,4,4,subsample = (2,2),init = 'uniform',activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = Convolution2D(64,3,3,subsample = (1,1),init = 'uniform',activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        z = Dense(128,init = 'uniform',activation = 'relu',name = 'ls_1')(x)
        ls = Dense(29,init = 'uniform',activation = 'relu',name = 'ls_2')(z)

        y = Dense(128,init = 'uniform',activation = 'relu',name = 'act_1')(x)
        Steering = Dense(1,activation = 'linear',init = 'uniform',name = 'act_2')(y)
        #Steering = Dense(1,weights = [np.random.uniform(-1e-8,1e-8,(512,1)),np.zeros((1,))], name='Steering')(lrn4)
        model = Model(S,[Steering,ls])
        adam = Adam(lr=0.000001,decay = 1e-6)
        model.compile(loss='mse', optimizer=adam)
        model.summary()
        return model, model.trainable_weights, S
def test(guide_name = None):
    from os.path import isdir
    from gym_torcs import TorcsEnv
    from ddpg import get_image
    '''
        Model preparation
        Retrieve all models and their placeholders here
    '''
    sess = tf.Session()
    guide,guide_weights,S = get_guide()
    if guide_name:
        # load the weights of the network in the given file
        # results of load_model(...) can not be used directly because the name there is not named
        with tf.device('/cpu:0'):
            temp = load_model(guide_name)
            guide.set_weights(temp.get_weights())
            # temp.save_weights('guide_weights.h5')
            # guide.load_weights('guide_weights.h5')
    critic,critic_action,critic_state = get_critic()
    if isdir('criticmodel.h5'):
        critic.load_weights('criticmodel.h5')
    '''
         Training Sequence
         The actor learns from the experiences
    '''
    #train_actor(guide)
    '''
        Start the testing sequence
        the low-dimension critic agent will be used
        actor takes high-dimension (64 x 64 x 12) input and output action
        part of the actor weights will be updated according to DDPG algorihm

    '''
    env = TorcsEnv(vision = True)
    ob = env.reset()
    l_s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
    img = get_image(ob)
    history = [img for i in range(4)]
    total_reward = 0
    num_episode = 5000
    step = 0
    optimizer = tf.train.AdamOptimizer(1e-6)
    sess.run(tf.initialize_all_variables())
    for epoch in range(num_episode):
        while True:
            s_t = np.concatenate(history,axis = -1)
            a_t,ls_t = guide.predict(s_t.reshape((1,) + s_t.shape))
            ob,reward,done,_ = env.step(a_t)
            l_s1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
            history.append(get_image(ob))
            history = history[1:]
            print("Step",step,"act",a_t,'reward',reward)
            total_reward = total_reward + reward
            step = step + 1
            '''
                DDPG: Actor Update
            '''
            #apply_ddpg_gradients(guide,critic.output,critic_action,critic_state,sess,optimizer,a_t,l_s.reshape((1,) + l_s.shape))
            if done:
                break
    print('total reward',total_reward)
def prepare_training_data(npys):
    X,y,z = zip(*npys)
    imgs = []
    for tensor in X:
        # frames: [(1,64,64,3)]
        frames = np.split(tensor,4,0)
        # frames: [(64,64,3)]
        frames = map(lambda img: img.squeeze(),frames)
        # imgs: [(64,64,12)]
        imgs.append(np.concatenate(frames,axis = -1))
    # X: (None,64,64,12)
    X = np.stack(imgs)
    y = np.array(y)
    z = np.array(z)
    return X,[y,z]

def train_critic(critic = None):
    from os import listdir
    from os.path import isfile
    from random import sample
#    if guide is not None and isfile('guide_model.h5'):
#        guide =load_model('guide_model.h5')
    if not critic:
        critic,A,S = get_critic()
    sess = tf.Session()
    with tf.name_scope('critic_loss'):
        loss_ph = tf.placeholder(tf.float32)
        loss_s = tf.summary.scalar('critic_loss',loss_ph)
    with tf.name_scope('critic_q'):
        q_ph = tf.placeholder(tf.float32)
        q_s = tf.summary.scalar('critic_q',q_ph)
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter('tmp/critic',graph = sess.graph)
    names = [('training/ls-%d.npy' % i,'training/a-%d.npy' % i,'training/r-%d.npy' % i) for i in range(99000)]
    vals = sample(names,20000)
    names = [n for n in names if n not in vals]
     
    def gen_sample(names):
        from random import shuffle
        shuffle(names)
        for ls,a,r in names:
            try:
                ls = np.load(ls).reshape(1,29)
                a = np.load(a)
                r = np.load(r).reshape(1,1)
                yield ([ls,a],r)
            except:
                continue
    tensorboard_callback = TensorBoard(log_dir='tmp/critic',write_graph = False,histogram_freq = None,write_images = True)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',factor = 0.1,patience = 20,min_lr = 0.00000001)

    critic.fit_generator(gen_sample(names),4,3000,gen_sample(vals),callbacks = [reduce_lr,tensorboard_callback])
    
    critic.save('guide_critic.h5')
def train_guide(guide = None):
    from os import listdir
    from os.path import isfile
    from random import sample
#    if guide is not None and isfile('guide_model.h5'):
#        guide =load_model('guide_model.h5')
    sess = tf.Session()
    with tf.name_scope('guide_loss'):
        loss_ph = tf.placeholder(tf.float32)
        loss_ls_ph = tf.placeholder(tf.float32)
        loss_s = tf.summary.scalar('guide_steering_loss',loss_ph)
        loss_ls = tf.summary.scalar('guide_low_dim_loss',loss_ls_ph)
    with tf.name_scope('guided_actor'):
        action_ph = tf.placeholder(tf.float32)
        action_s = tf.summary.scalar('guided_actor_action',action_ph)
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter('tmp/guide',graph = sess.graph)
    names = [('training/s-%d.npy' % i,'training/a-%d.npy' % i,'training/ls-%d.npy' % i) for i in range(100000)]
    vals = sample(names,20000)
    names = [n for n in names if n not in vals]

    # monitor the training
    last_loss = 99999999999
    last_loss_count = 0
    max_count = 1000
    for epoch in range(30000):
        # generic training
        batches = sample(names,16)
        batches = map(lambda data: (np.load(data[0]),np.load(data[1]).squeeze(),np.load(data[2])),batches)
        X,y = prepare_training_data(batches)
        loss = guide.train_on_batch(X,y)

        # sample some data and see if the network always gives the same action...
        act_sample = sample(names,16)
        acts = []
        X,y = prepare_training_data(batches)
        for x in X:
            x = x.reshape((1,) + x.shape)
            acts.append(guide.predict(x)[0][0])

#            a_summary = sess.run(action_s,feed_dict = {action_ph:act[0].squeeze()})
#            writer.add_summary(a_summary)

        # ...and write the summary to TensorBoard
        s = sess.run(summary_op,feed_dict = {loss_ph: loss[1],loss_ls_ph: loss[2],action_ph:np.average(acts)})
        writer.add_summary(s)
        # validation
        if epoch % 100 == 0:
            batches = sample(vals,64)
            batches = map(lambda data: (np.load(data[0]),np.load(data[1]).squeeze(),np.load(data[2])),batches)
            X,y = prepare_training_data(batches)
            val_loss = guide.test_on_batch(X,y)
            print("validation loss",val_loss)
        print ("epoch",epoch,"loss",loss)
    guide.save('guide_model.h5')
'''
    Util functions for the reinforcement learning loop
'''
def get_states(ob,history):
    img = get_image(ob)
    history.append(img)
    history = history[1:]

    img_states = np.concatenate(history,axis = -1)
    #img_states = img_states.reshape((1,) + img_states.shape)
    l_s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
    return img_states,l_s,history
def esp_process(a_t,esp):
    esp = 0. if esp < 0 else 1. if esp > 1. else esp # clamp epsilon
    should_random = (np.random.binomial(1,esp) == 1)
    return a_t + np.random.randn() if should_random else a_t

####
# return the weights of a trained critic based on the examples
####
def bp_critic(actor,critic,toy_critic,buff,sample_size = 16,gamma = 0.99):
    #0: get batch from buffer
    batch = buff.getBatch(sample_size)
    #1: extract data from batch
    img_s_t = np.array(map(lambda e: e[0][0],batch))
    s_t = np.array(map(lambda e: e[0][1],batch))
    a_t = np.array(map(lambda e: e[1],batch))
    img_s_t1 = np.array(map(lambda e: e[3][0],batch))
    s_t1 = np.array(map(lambda e: e[3][1],batch))
    dones = np.array(map(lambda e: e[4],batch))
    r = np.array(map(lambda e: e[2],batch))
    #2: form training data
    a_t1,ls_1 =  actor.predict(img_s_t1)
    a_t1 = np.array(a_t1)

    q_t1 = critic.predict([s_t1,a_t1])
    y = [r[i] + gamma * q_t1[i] if not dones[i] else r[i] for i in range(sample_size)]
    y = np.array(y)

    loss = toy_critic.train_on_batch([s_t,a_t],y)
 
    #4. get dq/da
    sess = K.get_session()
    S,A = toy_critic.input
    Q = toy_critic.output
    dq_da = tf.gradients(Q,A)
    grad = [sess.run(dq_da,feed_dict = {S: s_t[i].reshape((1,) + s_t[i].shape),A: a_t[i].reshape((1,) + a_t[i].shape)}) for i in range(sample_size)]
    # finally,return everything
    return toy_critic.get_weights(),grad,a_t,img_s_t,loss

def bp_actor(toy_actor,dq_da,a_t,s_t,optimizer):
    # compute da/dw: derivative of output actions w.r.t. to the weights involved.
    # TODO: debug me
    weights = get_action_trainable_weights(toy_actor)
    dq_da = np.array(dq_da).reshape(len(dq_da),1)

    da_dw = tf.gradients(toy_actor.output[0],weights,-dq_da)
    grads = zip(da_dw,weights)
    ag = optimizer.apply_gradients(grads)
    mag = [tf.norm(da) for da in da_dw]
    sess = K.get_session()
    sess.run(tf.global_variables_initializer())
    sess.run(ag,feed_dict = {toy_actor.input: s_t,toy_actor.output[0]: a_t})

def update_critic(critic,weights,tau):
    org_weights = critic.get_weights()
    for i,weight in enumerate(weights):
        org_weights[i] = tau * weight + (1 - tau) * org_weights[i]
    critic.set_weights(org_weights)

# TODO: test me
def update_actor(actor,weights,tau):
    org_weights = actor.get_layer('act_1').get_weights() + actor.get_layer('act_2').get_weights()
    for i,weight in enumerate(weights):
        org_weights[i] = tau * weight + (1 - tau) * org_weights[i]
    actor.get_layer('act_1').set_weights(map(np.array,org_weights[0:2]))
    actor.get_layer('act_2').set_weights(map(np.array,org_weights[2:4]))

    
def reinforce(guide,critic):
    from gym_torcs import TorcsEnv
    from ReplayBuffer import ReplayBuffer
    
    env = TorcsEnv(vision = True,throttle = False,gear_change = False)
    buff = ReplayBuffer(100000)
    esp = 0.
    tau = 0.001
    gamma = 0.99
    ob = env.reset()
    toy_critic,CA,CS = get_critic()
    toy_critic.set_weights(critic.get_weights())
    toy_actor,AW,AS = get_guide()
    toy_actor.set_weights(guide.get_weights())

    # ddpg computation with tf optimizer
    optimizer = tf.train.AdamOptimizer(0.00001)

    ################################ main loop #########################
    for epoch in range(10000):
        history = [get_image(ob) for i in range(4)]
        total_reward = 0
        for step in range(10000):
            img_states,low_states,history = get_states(ob,history)
            a_t,pls_t = guide.predict(img_states.reshape((1,) + img_states.shape))

            a_t = esp_process(a_t,esp)
            esp = esp * 0.993

            ob,reward,done,_ = env.step(a_t[0])
            total_reward += reward
            img_states_1, low_states_1,history = get_states(ob,history)
            # add this experience to buffer
            buff.add((img_states,low_states),a_t[0],reward,(img_states_1,low_states_1),done)

            if buff.num_experiences < 16: continue
            #train critic
            critic_weights,dq_da,batch_a_t,batch_img_s_t,loss = bp_critic(guide,critic,toy_critic,buff,16,gamma)
            #train actor
            #TODO: me
            bp_actor(toy_actor,dq_da,batch_a_t,batch_img_s_t,optimizer)

            # extract weights of the toy actor for actor weight updating
            actor_weights = get_action_trainable_weights(toy_actor,True)

            #update critic and actor
#            update_critic(critic,critic_weights,tau)
#            #TODO: me
#            update_actor(guide,actor_weights,tau)
            print 'epoch %d, step %d' % (epoch,step)
            if done:
                ob = env.reset(relaunch = (epoch % 3 == 0))
                break
        print 'epoch:%d, reward: %d' % (epoch,total_reward)


if __name__ == '__main__':
    org_guide = load_model('guide_model.h5')
    guide,weights,S = get_guide()
    guide.set_weights(org_guide.get_weights())
    critic = load_model('guide_critic.h5')
    reinforce(guide,critic)

    #train_critic(None)
    #test('guide_model.h5')
