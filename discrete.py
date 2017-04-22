from model import *
from gym_torcs import TorcsEnv
from ReplayBuffer import ReplayBuffer
from utils import *
from collections import deque
import tensorflow as tf
def discrete_reinforce():
	# initialization
	tf.python.control_flow_ops = tf
	env = TorcsEnv(vision = True,throttle = False,gear_change = False)
	buff = ReplayBuffer(100000)
	history = deque()
	
	writer = tf.summary.FileWriter('tmp/discrete')

	# env. initialization
	ob = env.reset()

	esp = 1.
	lr = 0.0001
	gamma = 0.99
	tau = 0.01
	batch_size = 32
	max_epoch = 10000
	max_steps = 10000

	num_output = 5

	# get model
	actor = low_guide_v1(lr,num_output)
	target_actor = low_guide_v1(lr,num_output)
	transfer_weights(actor,target_actor)

	# summary ops and phs
	reward_ph = tf.placeholder(tf.float32)
	loss_ph = tf.placeholder(tf.float32)
	q_ph = tf.placeholder(tf.float32)
	#target_q = tf.placeholder(tf.float32,[batch_size,num_output])

	reward_summary = tf.summary.scalar('reward',reward_ph)
	loss_summary = tf.summary.scalar('loss',loss_ph)
	q_summary = tf.summary.scalar('estimated_q',q_ph)

	# gradient inspection
	grads = tf.gradients(actor.output,actor.trainable_weights)
	grad_summary = [tf.summary.histogram('bp_grad-%d' % i,g) for (i,g) in enumerate(grads)]
	grad_summary = tf.summary.merge(grad_summary)

	for epoch in range(max_epoch):
		history = deque()
		[history.append(get_low_states(ob)) for i in range(4)]
		total_reward = 0
		total_loss = 0
		for step in range(max_steps):
			st = get_states(ob,history,False) # interpret and prepare the states,*** st is a stacked states***
			act = actor.predict(st.reshape((1,) + st.shape)) # ask for action
			estimated_max_q = np.max(act)

			# input processing and step
			act = get_inferred_steering(act,num_output) # convert the discrete decision to continuous
			act,is_org = discrete_esp_process(esp,act,num_output)
			esp -= 1./10000

			ob,reward,done,_ = env.step([act]) # execute and observe
			
			# post step proessing
			total_reward += reward
			st1 = get_states(ob,history,False)
			# post observation
			buff.add(st,act,reward,st1,done,step) # add experience
			# training
			if step < batch_size: continue
			experiences = buff.getBatchMixed(batch_size)
			
			X,y,a_t = preprocess_batch(experiences,actor,gamma,batch_size,target_actor)
			y = prepare_label(X,a_t,y,actor,num_output)
			loss = actor.train_on_batch(X,y)
			total_loss += loss

			update_network(actor,target_actor,tau)

			# logging and stats
			print 'Epoch: %d, Step: %d, Act: %f, Loss: %f,AI: %s' % (epoch,step,act,loss,str(is_org))
			writer.add_summary(K.get_session().run(loss_summary,feed_dict = {loss_ph:loss}))
			writer.add_summary(K.get_session().run(q_summary,feed_dict = {q_ph:estimated_max_q}))
			writer.add_summary(K.get_session().run(grad_summary,feed_dict = {actor.input: X}))
			# termination condition
			if done:
				ob = env.reset(epoch % 3 == 1)
				break


		print '************ Epoch %d : Reward %f ************' % (epoch,total_reward)
		# post epoch stuff
		if epoch % 10 == 0:
			actor.save_weights('low_guide_v1_weights.h5')
		# epoch summaries
		writer.add_summary(K.get_session().run(reward_summary,feed_dict = {reward_ph:total_reward}))

if __name__ == '__main__':
	discrete_reinforce()


''