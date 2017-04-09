from os.path import isfile
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from proportional import Experience
from encoder import get_pretrained_encoder
from ddpg import get_image
import keras.backend as K
from collections import deque
tf.python.control_flow_ops = tf
K.set_learning_phase(0)
def get_critic(weight_files = None,LRC = 0.001):

        HIDDEN1_UNITS = 300
        HIDDEN2_UNITS = 600
        S = Input(shape=[29])
        A = Input(shape=[1],name='action2')
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A)
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = merge([h1,a1],mode='sum')
        h3 = Dense(HIDDEN1_UNITS, activation='relu')(h2)
        V = Dense(1,activation='linear')(h3)
        model = Model(input=[S,A],output=V)
        adam = Adam(lr = LRC)
        model.compile(loss='mse', optimizer=adam)
        if weight_files:
            model.load_weights(weight_files)
#        K.get_session().run([adam.beta_1.initializer,adam.beta_2.initializer])
        return model, A, S

def get_action_trainable_weights(guide,isValue = False):
    return guide.get_weights() if isValue else guide.trainable_weights
#    if not isValue:
#        return guide.get_layer('act_1').trainable_weights + guide.get_layer('act_2').trainable_weights
#    else:
#        return guide.get_layer('act_1').get_weights() + guide.get_layer('act_2').get_weights()

def add_noise(action):
    return action + np.random.randn(action.shape)
def get_guide(weight_files = None):
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
def test(guide_weight):
    from os.path import isdir
    from gym_torcs import TorcsEnv
    from ddpg import get_image
    '''
        Model preparation
        Retrieve all models and their placeholders here
    '''
    sess = tf.Session()
    guide,guide_weights,S = get_guide(guide_weight)
    
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
            a_t = guide.predict(s_t.reshape((1,) + s_t.shape))
            ob,reward,done,_ = env.step(a_t)
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
    return X,y

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
    vals = sample(names,10000)
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
def train_guide(guide = None,weight_name = None):
    from os import listdir
    from os.path import isfile
    from random import sample
#    if guide is not None and isfile('guide_model.h5'):
#        guide =load_model('guide_model.h5')
    sess = tf.Session()
    with tf.name_scope('guide_loss'):
        loss_ph = tf.placeholder(tf.float32)
        #loss_ls_ph = tf.placeholder(tf.float32)
        loss_s = tf.summary.scalar('guide_steering_loss',loss_ph)
        #loss_ls = tf.summary.scalar('guide_low_dim_loss',loss_ls_ph)
#    with tf.name_scope('guided_actor'):
#        action_ph = tf.placeholder(tf.float32)
#        action_s = tf.summary.scalar('guided_actor_action',action_ph)
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter('tmp/guide',graph = sess.graph)
    names = [('training/s-%d.npy' % i,'training/a-%d.npy' % i,'training/ls-%d.npy' % i) for i in range(100000) if isfile('training/s-%d.npy' % i)]
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
            acts.append(guide.predict(x)[0])

        # ...and write the summary to TensorBoard
        s = sess.run(summary_op,feed_dict = {loss_ph: loss})
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
    if weight_name:
        guide.save_weights(weight_name)

'''
    Util functions for the reinforcement learning loop
'''
def get_states(ob,history):
    img_states = np.concatenate(history,axis = -1)

    img = get_image(ob)
    history.append(img)
    history.popleft()
    
    #img_states = img_states.reshape((1,) + img_states.shape)
    l_s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
    return img_states,l_s
def guided_action(ob):
    res = ob[4] * 3
    res -= ob[8] * .09
    return [[res]]
def esp_process(ob,a_t,esp):
    esp = 0. if esp < 0 else 1. if esp > 1. else esp # clamp epsilon
    ran = np.random.binomial(2,esp)
    should_not_guide = (ran == 0)
    should_random = (ran == 2)
    return (a_t + np.random.normal(0,0.2),True) if should_random else (a_t,False) if should_not_guide else (guided_action(ob),None) # None for guided

####
# return the weights of a trained critic based on the examples
####
def bp_critic(actor,critic,toy_critic,buff,dq_da_op,sample_size = 16,gamma = 0.99,loss_ph = None,grad_summary = None):
    #0: get batch from buffer
    batch = buff.getBatchMixed(sample_size)
    #1: extract data from batch
    img_s_t = np.array(map(lambda e: e[0][0],batch))
    s_t = np.array(map(lambda e: e[0][1],batch))
    a_t = np.array(map(lambda e: e[1],batch))
    img_s_t1 = np.array(map(lambda e: e[3][0],batch))
    s_t1 = np.array(map(lambda e: e[3][1],batch))
    dones = np.array(map(lambda e: e[4],batch))
    r = np.array(map(lambda e: e[2],batch))
    #2: form training data
    #a_t1,ls_1 =  actor.predict(img_s_t1)
    a_t1 = actor.predict(img_s_t1)
    a_t1 = np.array(a_t1)

    q_t1 = critic.predict([s_t1,a_t1])
    y = [r[i] + gamma * q_t1[i] if not dones[i] else r[i] for i in range(sample_size)]
    y = np.array(y)

    loss = toy_critic.train_on_batch([s_t,a_t],y)

    #4. get dq/da
    sess = K.get_session()
    #grad = [sess.run(dq_da_op,feed_dict = {S: s_t[i].reshape((1,) + s_t[i].shape),A: a_t[i].reshape((1,) + a_t[i].shape)}) for i in range(sample_size)]
    grad = sess.run(dq_da_op,feed_dict = {S: s_t,A:a_t})
    #TODO: why is this 16 instead of 1? grad = np.mean(grad) # 1/N * ...

    # finally,return everything
    return toy_critic.get_weights(),grad,a_t,img_s_t,loss

def bp_actor(toy_actor,dq_da,a_t,s_t,ag,dq_da_ph):
#    weights = get_action_trainable_weights(toy_actor)
    dq_da = np.array(dq_da).reshape(16,1)
#    mag = [tf.norm(da) for da in da_dw]
    print dq_da,dq_da.shape
    sess = K.get_session()
    sess.run(ag,feed_dict = {
        toy_actor.input: s_t,
        toy_actor.output: a_t,
        dq_da_ph: dq_da
    })

def update_critic(critic,weights,tau):
    org_weights = critic.get_weights()
    for i,weight in enumerate(weights):
        org_weights[i] = tau * weight + (1 - tau) * org_weights[i]
    critic.set_weights(org_weights)

def update_actor(actor,weights,tau):
    org_weights = actor.get_weights()
    for i,weight in enumerate(weights):
        org_weights[i] = tau * weight + (1 - tau) * org_weights[i]
    actor.set_weights(org_weights)

    
def reinforce(guide,toy_actor,critic,toy_critic):
    from gym_torcs import TorcsEnv
    from ReplayBuffer import ReplayBuffer
    # moving average helper function
    def moving_average(a, n=3):
        return np.convolve(a,[1./n for i in range(n)],mode = 'valid')

    # logging variables
    with tf.name_scope('ddpg'):
        reward_ph = tf.placeholder(tf.float32,())
        reward_sum= tf.summary.scalar('reward',reward_ph)
    logger = tf.summary.FileWriter('tmp/reinforce',graph = K.get_session().graph)
    # env variables
    env = TorcsEnv(vision = True,throttle = False,gear_change = False)
    buff = ReplayBuffer(100000)
    #buff = Experience(100000,16,0.4)
    esp = 1.
    tau = 0.001
    gamma = 0.99
    LRA = 0.000001
    LRC = 0.00001
    esp_window_size = 10
    ob = env.reset()
    reward_history = []
    step_history = [0 for i in range(esp_window_size * 2)] # pad history so that error will not arise when evaluating esp.

    #bp_critic common variables
    S,A = toy_critic.input
    Q = toy_critic.output
    dq_da_op = tf.gradients(Q,A)
    #bp_actor common variables: ddpg
    # get weights variables as actor
    actor_weights = get_action_trainable_weights(toy_actor)
    # prepare placeholders for two 
    dq_da_ph = tf.placeholder(tf.float32,shape = (16,1)) # action dim + (1,)
    da_dw = tf.gradients(toy_actor.output,actor_weights,-dq_da_ph,name = 'da_dw')
    # apply gradients to actor weights
    grads = zip(da_dw,actor_weights)
    ag = tf.train.AdamOptimizer(LRA).apply_gradients(grads)
    initialize_optimizer_variables()

    # logdir 2: record the actor weights and the gradient weights
    with tf.name_scope('critic_loss'):
        critic_loss_ph = tf.placeholder(tf.float32,())
        loss_summary_ops = tf.summary.scalar('loss',critic_loss_ph)
    with tf.name_scope('actor_weights'):
        aw_summary_ops = [tf.summary.histogram(('weights-%d' % i) if i % 2 == 0 else ('bias-%d' % (i - 1)), x ) for (i,x) in enumerate(actor_weights)]
        #aw_summary_ops = [tf.summary.image('weights-%d' % (i + 1),tf.expand_dims(tf.expand_dims(x,-1),0)) if i == 0 else tf.summary.histogram('bias-%d' % (i + 1),x) for (i,x) in enumerate(actor_weights)]
    
#    with tf.name_scope('gradients'):
#        grad_summary_ops = [tf.summary.image('weights-%d' % (i + 1),tf.expand_dims(tf.expand_dims(x,-1),0)) if i == 0 else tf.summary.histogram('bias-%d' % (i + 1),x) for (i,x) in enumerate(da_dw)]

    # logdir 3: embedding visualization
    import os
    summary_op = tf.summary.merge(aw_summary_ops + [reward_sum,loss_summary_ops])
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = guide.get_layer('act_1').output.name
    # ddpg computation with tf optimizer
    ################################ main loop #########################
    for epoch in range(10000):
        #################### epoch start: preprocessing ################
        # history preparation, variable initialization, exploration planning
        history = deque([get_image(ob) for i in range(4)])
        total_reward = 0.
        last_s_avg = 200.0
        a = 0.001
        b = 0.8
        c = 1.
        s_avg = moving_average(step_history,n = esp_window_size)[-1] # get the max average
        # esp criteria:
        # esp should raise to "b" starting from "a", when the step reaches the moving average of the maximum steps.
        # The formula would be esp = a * exp(b - ln (b/a)(s' - s)), where s' is the moving average of max steps,
        # s is the current step
        get_esp = lambda s: a * np.exp((s / s_avg) ** c * np.log(b/a)) if s < s_avg else b
        ################### start looping ###############################
        for step in range(10000):
            img_states,low_states = get_states(ob,history)
            a_t = guide.predict(img_states.reshape((1,) + img_states.shape))
            
            np.save('s-%d-%d.npy' % (epoch,step),np.concatenate(history,axis = -1))
            # get exploration propability and apply e-greedy
            action_esp = get_esp(step)
            a_t,is_random= esp_process(ob,a_t,action_esp)

            # step and get observation
            ob,reward,done,_ = env.step(a_t[0])
            total_reward += reward
            img_states_1, low_states_1 = get_states(ob,history)
            # add this experience to buffer
            buff.add((img_states,low_states),a_t[0],reward,(img_states_1,low_states_1),done,step)

            if buff.num_experiences < 16: continue
            #train critic
            critic_weights,dq_da,batch_a_t,batch_img_s_t,loss = bp_critic(guide,critic,toy_critic,buff,dq_da_op,16,gamma)
            #train actor
            bp_actor(toy_actor,dq_da,batch_a_t,batch_img_s_t,ag,dq_da_ph)

            # extract weights of the toy actor for actor weight updating
            actor_weights = get_action_trainable_weights(toy_actor,True)

            #update critic and actor
            update_critic(critic,critic_weights,tau)
            update_actor(guide,actor_weights,tau)

            # embedding visualization
            if step % 5 == 0:
                projector.visualize_embeddings(logger,config)
            print 'epoch %d, step %d, act_exp %f,is random %s' % (epoch,step,action_esp,'guide' if is_random is None else 'AI' if not is_random else 'random')
            if done:
                ob = env.reset(relaunch = (epoch % 3 == 0))
                break
        print 'epoch:%d, reward: %f' % (epoch,total_reward)

        # save the weights every 10 epochs
        if epoch % 10 == 0:
            guide.save_weights('guided_actor_weights_trained.h5')
            critic.save_weights('guided_critic_weights_trained.h5')
        # end of epoch, updating stuff ,logging and cleaning up
        # logging
        summary = K.get_session().run(summary_op,feed_dict = {reward_ph: total_reward,critic_loss_ph: loss})
        logger.add_summary(summary)
        reward_history.append(total_reward)

        step_history.append(step)
        
def initialize_optimizer_variables():
    opt_vars = [v for v in tf.global_variables() if 'beta' in v.name or 'Adam' in v.name]
    K.get_session().run(tf.variables_initializer(opt_vars))
def test_critic(weight_file):
    guide,a,b = get_guide(weight_file)

if __name__ == '__main__':
    guide,weights,S = get_guide()
    toy_guide,weights,S = get_guide()
    critic,A,S = get_critic()
    toy_critic,A,S = get_critic()
#    # logger creation and config
    reinforce(guide,toy_guide,critic,toy_critic)
#
    #train_critic(None)
#    K.get_session().run(tf.global_variables_initializer())
#    train_guide(guide)
    #test('guided_actor_weights_trained.h5')
