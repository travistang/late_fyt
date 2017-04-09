from model import *
from gym_torcs import TorcsEnv
from ReplayBuffer import ReplayBuffer
from utils import *
from collections import deque
def discrete_reinforce():
	# initialization
	env = TorcsEnv(vision = True,throttle = False,gear_change = False)
	buff = ReplayBuffer(10000)
	history = deque()
	# env. initialization
	ob = env.reset()

	esp = 1.
	lr = 0.001
	gamma = 0.99
	batch_size = 32
	max_epoch = 10000
	max_steps = 10000
	# get model
	actor = low_guide_v1(lr)

	for epoch in range(max_epoch):
		history = deque()
		[history.append(get_low_states(ob)) for i in range(4)]
		total_reward = 0
		for step in range(max_steps):
			st = get_states(ob,history,False) # interpret and prepare the states,*** st is a stacked states***
			act = actor.predict(st.reshape((1,) + st.shape)) # ask for action
			act = get_inferred_steering(act) # convert the discrete decision to continuous
			st1,reward,done,_ = env.step(act) # execute and observe
			total_reward += reward
			st1 = get_states(ob,history,False)
			# post observation
			buff.add(st,act,reward,st1,done,step) # add experience
			# training
			if step < batch_size: continue
			experiences = buff.getBatch(batch_size)
			
			X,y = preprocess_batch(experiences,actor,gamma,batch_size)
			loss = actor.train_on_batch(X,y)

			print 'Epoch: %d, Step: %d, Act: %f, Loss: %f' % (epoch,step,act,loss)
		print '************ Epoch %d : Reward %f ************' % (epoch,total_reward)

if __name__ == '__main__':
	discrete_reinforce()


