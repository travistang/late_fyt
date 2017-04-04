from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import *
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam,RMSprop
import tensorflow as tf
from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBuffer
from proportional import Experience
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
import tensorflow as tf
import cv2
OU = OU()       #Ornstein-Uhlenbeck Process
tf.python.control_flow_ops = tf
colors = [np.random.uniform(0,1,(1,3)) for i in range(20)]
def get_triple_actions(client,steering):
    return np.hstack(steering,client.R.d['accel'],0.)
def get_image(ob):
    img = ob[-1]
    vision = ob[-1]
    img = np.array(vision).reshape(64,64,3)
    img = np.flipud(img).astype(np.float32)/255.0 # normalized, colored image
    #img = cv2.cvtColor(np.flipud(img).astype(np.float32),cv2.COLOR_BGR2GRAY)/255.0
    return img

def get_sample(sess,env,sample_size,buff):
    history = []
    ls_old = None
    for z in range(sample_size):
        print "Gathering %d/%d sample..." % (z,sample_size)
        act = np.random.normal(0,0.7,(1,1))
        ob,r,done,_ = env.step(act)
        ls = np.hstack((ob.angle,ob.track,ob.trackPos,ob.speedX,ob.speedY,ob.speedZ,ob.wheelSpinVel/100.0,ob.rpm))
        if ls_old is None: ls_old = ls
        img = get_image(ob)
        history.append(img)
        # stack histories to form states
        if len(history) > 4:
            s_t = np.concatenate(history[0:4],axis = -1)
            s_t1 = np.concatenate(history[1:],axis = -1)
            history = history[1:]
            buff.add((s_t,ls_old),act,r,(s_t1,ls),done)
            ls_old = ls
        if done:
            ob = env.reset()
            ls_old = None
            history = []

def train_critic(sess,critic,actor,buff):
    nb_epoch = 50
    batch_size = 16
    for i in range(nb_epoch):
        batch,weights,indices = buff.select(0)
        #  extract things from batch
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = np.asarray([e[1] for e in batch])
        a = actor.target_model.predict(new_states)
        #a = a.reshape(a.shape + (1,))
        target_q_values = critic.target_model.predict([new_states, a])
        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + GAMMA*target_q_values[k]

        loss += critic.model.train_on_batch([states,actions],y_t)
        critic_prediction = critic.model.predict([states,actions],BATCH_SIZE)

        loss_per_sample = sess.run(tf.abs(tf.subtract(critic_prediction, y_t))) # evaluating the loss of critics per sample
        #buff.priority_update(indices,loss_per_sample) # because the priority is the absolute difference of the error
    critic.target_train()
def playGame(train_indicator=1):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 16
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.  #Learning rate for Actor
    LRC = 0.000001     #Lerning rate for Critic

    action_dim = 1  #Steering only!
    state_dim = 29  #of sensors input

    np.random.seed(1337)

    vision = True

    EXPLORE = 10000.
    episode_count = 20000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    count = 0
    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_learning_phase(0)
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    guide = load_model('guide_model.h5')
    actor.model.set_weights(guide.get_weights())
    actor.target_model.set_weights(guide.get_weights())
    #buff = Experience(BUFFER_SIZE,BATCH_SIZE,0.6)    #Create replay buffer
    buff = ReplayBuffer(BUFFER_SIZE)
    # *************************************** summary section *************************************#
    summaries = []
    with tf.name_scope('reward'):
        reward_ph = tf.placeholder(tf.float32)
        s = tf.summary.scalar('average reward',reward_ph)
        summaries.append(s)
    with tf.name_scope('image_input'):
        img_input = tf.placeholder(tf.float32,(1,64,64,3))
        s = tf.summary.image('image_input',img_input)
        summaries.append(s)
    with tf.name_scope('estimated_q_value'):
        q_loss = tf.placeholder(tf.float32)
        estimate_q = tf.placeholder(tf.float32)
        a_out = tf.placeholder(tf.float32)
        sq = tf.summary.scalar('estimated Q value',estimate_q)
        sl = tf.summary.scalar('Q loss', q_loss)
        sa = tf.summary.scalar('Action', a_out)
        summaries = summaries + [sl,sa]
    writer = tf.summary.FileWriter('tmp/actor',graph = sess.graph)
    summary_op = tf.summary.merge(summaries)
    # ***************************************/summary section *************************************#
    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=False,gear_change=False)

    print("TORCS Experiment Start.")
    # random search to populate buffer
    ob = env.reset()
    #from encoder import get_encoder
    history = []
    summary_history = []
    img_history = []
    sample_size = 100 # WARNING: at least  BATCH_SIZE
    get_sample(sess,env,sample_size,buff)
    '''
        Training begins
    '''
    for i in range(episode_count):
        history = []
        print("Episode : " + str(i) + " Replay Buffer " + str(buff.size()))

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        img = get_image(ob)
        history = [img for k in range(4)] # populate the history
        #s_t = np.stack(history,axis = -1)
        s_t = np.concatenate(history,axis = -1)
        ls_t = np.hstack((ob.angle,ob.track,ob.trackPos,ob.speedX,ob.speedY,ob.speedZ,ob.wheelSpinVel/100.0,ob.rpm))
        total_reward = 0.
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            a_t_original, _ = actor.target_model.predict(s_t.reshape((1,) + s_t.shape))
            # plotting stuff:
            count = count + 1

            #noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
#            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
#            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)
            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
#            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
#            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            ob, r_t, done, info = env.step(a_t[0])
            client = env.get_client()
            #la_t = get_triple_actions(client,a_t[0])
            ls_t1 = np.hstack((ob.angle,ob.track,ob.trackPos,ob.speedX,ob.speedY,ob.speedZ,ob.wheelSpinVel/100.0,ob.rpm))
            q = critic.model.predict([ls_t1.reshape(1,29),a_t]).squeeze()
            img = get_image(ob)
            history.append(img)
            history = history[1:]
            s_t1 = np.concatenate(history,axis = -1)
            buff.add((s_t,ls_t), a_t[0], r_t, (s_t1,ls_t1), done)      #Add replay buffer
#            p_t = 1.
#            if buff.tree.filled_size() > 0:
#                p_t = max([buff.tree.get_val(z) for z in range(buff.tree.filled_size())]) # assign the new tuple with the highest priority
#            buff.add((s_t,a_t[0],r_t,s_t1,done), p_t)
#            print('p_t',p_t)
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            #batch,batch_weights,indices = buff.select(0.4) # beta = 0.4
            states = np.asarray([e[0][1] for e in batch])
            img_states = np.asarray([e[0][0] for e in batch]) # new states
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3][1] for e in batch])
            new_img_states = np.asarray([e[3][0] for e in batch]) # new states
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])
            print "shape of y_t: %s" % str(y_t.shape)
            a,ls = actor.target_model.predict(new_img_states)
            #a = a.reshape(a.shape + (1,))
            target_q_values = critic.target_model.predict([new_states, a])
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
            grads = 0
            a_grad = 0
            actor_summary = None
            if (train_indicator and len(history) >= 4):
                actions = actions.reshape(BATCH_SIZE,1)
                y_t = y_t.reshape(BATCH_SIZE,1)
                print "action shape: %s, y_t shape: %s" % (actions.shape,y_t.shape)
                loss += critic.model.train_on_batch([states,actions],y_t)
                #critic_prediction = critic.model.predict([states,actions],BATCH_SIZE)

                #loss_per_sample = sess.run(tf.abs(tf.subtract(critic_prediction, y_t))) # evaluating the loss of critics per sample
                #buff.priority_update(indices,loss_per_sample) # because the priority is the absolute difference of the error
                '''
                    Training part...
                a_for_grad = actor.model.predict(img_states)
                grads = critic.gradients(states, a_for_grad) # grads = grad_tc(Q)
                a_grad = actor.train(img_states, grads.reshape(BATCH_SIZE,1)) # a_grad = policy gradient
                actor_summary = actor.log(img_states,grads)
                actor.target_train()
                critic.target_train()

                '''
            total_reward += r_t
            s_t = s_t1
            ls_t = ls_t1
            # logging stuff
            print("Episode", i, "Step", step, "act",a_t_original,"act_noise", a_t, "Loss", loss)

            summary = sess.run(summary_op,feed_dict = {
                reward_ph: total_reward,
                q_loss: loss,
                estimate_q:q,
                img_input: s_t[:,:,-3:].reshape(1,64,64,3),
                a_out: a_t_original[0][0]
            })
            writer.add_summary(actor_summary)
            writer.add_summary(summary)
            step += 1
            if done:
                history = []
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

        # TODO: put all required values to feed_dict!

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()
