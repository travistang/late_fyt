import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.optimizers import Adam
from keras.callbacks import *
from keras import backend as K
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor,MultiInputProcessor
from gym_torcs import TorcsEnv
import tensorflow as tf
from model import *
from utils import get_low_states,get_image
from rl.callbacks import *
from collections import deque
def main(cnn_type,num_filter,hist_length):
	ENV_NAME = 'Torcs'

	K.set_learning_phase(0)
	# file naming
	checkpoint_format = '{}_{}_{}_{{step}}.h5'.format(cnn_type,num_filter,hist_length)
	csv_filename = "{}_{}_{}.csv".format(cnn_type,num_filter,hist_length)
	log_filename = "{}_{}_{}.log".format(cnn_type,num_filter,hist_length)
	weights_filename = "{}_{}_{}.h5f".format(cnn_type,num_filter,hist_length)
	class HistoryContainer():
		def __init__(self,history_length = 4):
			self.history = deque()
			self.history_length = history_length
			self.step = 0
		def add(self,state):
			# if the history deque is empty, fill it with the given state
			if len(self.history) == 0: [self.history.append(state) for _ in range(self.history_length)]
			else: self.history.append(state) # normal append
			
			self.step += 1

			# remove excess states
			while len(self.history) > self.history_length: self.history.popleft()
		def get(self):
			assert len(self.history) > 0
			return np.concatenate(self.history,axis = -1)

		def get_stack(self):
			assert len(self.history) > 0
			return np.stack(self.history,axis = 0)

		def empty(self):
			self.history.clear()

	class MyProcessor(Processor):
		def __init__(self,history_obj,fork = False):
			self.historyManager = history_obj
			self.fork = fork
		def process_observation(self,ob):
			img = get_image(ob)
			self.historyManager.add(img)
			obs = self.historyManager.get() if not self.fork else self.historyManager.get_stack()
			#obs = self.historyManager.get_stack()
			return obs

	gym.undo_logger_setup()
	tf.python.control_flow_ops = tf

	# Get the environment and extract the number of actions.
	env = TorcsEnv(vision = True,throttle = False)
	np.random.seed(123)
	assert len(env.action_space.shape) == 1
	nb_actions = 1
	# Next, we build a very simple model.
	if cnn_type == 'stack':
		actor,critic = stack_model(num_action = nb_actions,hist_len = hist_length,num_filters = num_filter)
	elif cnn_type == 'fork':
		actor,critic = fork_model(num_action = nb_actions,hist_len = hist_length,num_filters = num_filter)
	else:
		actor,critic = LSTM_model(num_action = nb_actions,hist_len = hist_length,num_filters = num_filter)
	histMgr = HistoryContainer(hist_length)
	processor = MyProcessor(histMgr,cnn_type in ["fork","LSTM"])
	# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
	# even the metrics!
	memory = SequentialMemory(limit=100000, window_length=1)
	random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
	agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=critic.input[0],
	                  memory=memory, nb_steps_warmup_critic=500, nb_steps_warmup_actor=2000,
	                  random_process=random_process, gamma=.99, target_model_update=1e-3,
	                  processor = processor)
	agent.compile(Adam(lr=.0001, clipnorm=1., decay=1e-4), metrics=['mae'])

	callbacks = [FileLogger('ddpg.log')]
	callbacks.append(ModelIntervalCheckpoint(checkpoint_format,interval = 5000))
	callbacks.append(CSVLogger(csv_filename))

	# Okay, now it's time to learn something! We visualize the training here for show, but this
	# slows down training quite a lot. You can always safely abort the training prematurely using
	# Ctrl + C.
	agent.fit(env, nb_steps=40000, visualize=False, verbose=1, nb_max_episode_steps=10000,callbacks = callbacks)

	# After training is done, we save the final weights.
	agent.save_weights(weights_filename, overwrite=True)

	# Finally, evaluate our algorithm for 5 episodes.
	#agent.test(env, nb_episodes=3, visualize=False, nb_max_episode_steps=10000)

if __name__ == '__main__':
	from itertools import product
	#cnn_types = ['stack','fork']
	#num_filters = [16,32]
	hist_lens = [4,6,8,12]
	cnn_types = ['LSTM']
	num_filters = [32]

	for cnn_t, nf, hist_len in product(cnn_types,num_filters,hist_lens):
		main(cnn_t, nf, hist_len)