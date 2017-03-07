from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam,RMSprop
import tensorflow as tf
from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
OU = OU()       #Ornstein-Uhlenbeck Process
tf.python.control_flow_ops = tf
plt.ion()
colors = [np.random.uniform(0,1,(1,3)) for i in range(20)]
def get_image(ob):
    img = ob[-1]
    vision = ob[-1]
    img = np.array(vision).reshape(64,64,3)
    img = cv2.cvtColor(np.flipud(img).astype(np.float32),cv2.COLOR_BGR2GRAY)/255.0
    return img.reshape((64,64,1))
    #return img

def playGame(train_indicator=1):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.1     #Target Network HyperParameters
    LRA = 0.00001    #Learning rate for Actor
    LRC = 0.00000001     #Lerning rate for Critic

    action_dim = 1  #Steering only!
    state_dim = 29  #of sensors input

    np.random.seed(1337)

    vision = True

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 500
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0
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
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=False,gear_change=False)

    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("TORCS Experiment Start.")
    # random search to populate buffer
    ob = env.reset()
    #from encoder import get_encoder
    #encoder = get_encoder()
    history = []
    img_history = []
    qs = []
    sample_size = 40
    for i in range(sample_size):
        print "Gathering %d/%d sample..." % (i,sample_size)
        act = np.random.normal(0,0.7,(1,))
        ob,r,done,_ = env.step(act)
        img = get_image(ob)
        np.save('%d.npy' % i, img)
        img_history.append(img)
        history.append(img)
        # stack histories to form states
        if len(history) > 4:
            s_t = np.stack(history[0:4])
            s_t1 = np.stack(history[1:])
            history = history[1:]
            buff.add(s_t,act,r,s_t,done)
        if done:
            ob = env.reset()
            history = []
    # train encoder and save the weights
    #encoder.compile(optimizer = 'adam',nb_epoch = 20,batch_size = 16)
    for i in range(episode_count):
        qs = []
        '''
        plot.set_xdata(range(len(qs)))
                plot.set_ydata(qs)
                plt.draw()
        '''
        history = []
        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        img = get_image(ob)
        [history.append(img) for k in range(4)] # populate the history
        #s_t = np.stack(history,axis = -1)
        s_t = np.stack(history)
        total_reward = 0.
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            a_t_original = actor.model.predict(s_t.reshape((1,) + s_t.shape))
            # plotting stuff:
            '''
            for ind,norm in enumerate(actor.get_bias_norm()):
                plt.scatter(count,norm,c = colors[ind])
            '''
            count = count + 1
            #plt.pause(0.005)
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
#            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
#            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)
            # TODO: what if we sample actions randomly?
            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
#            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
#            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            ob, r_t, done, info = env.step(a_t[0])

            #img = np.ndarray((64,64,3))
            img = get_image(ob)
            history.append(img)
            history = history[1:]
            #s_t1 = np.stack(history,axis = -1)
            s_t1 = np.stack(history)
            np.save("%d.npy" % count, img)
            #buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            buff.add(s_t,a_t,r_t,s_t1,done)
            #Do the batch update
            batch = buff.getBatchGreedy(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
            qs.append(target_q_values[0])
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]

            if (train_indicator and len(history) >= 4):
                loss += critic.model.train_on_batch([states,actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)

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

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()
