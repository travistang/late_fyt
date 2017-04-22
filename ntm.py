import tensorflow as tf
import numpy as np


class NTM(object):
	def __init__(self,session, mem_size, mem_dim,controller):
		self.sess = session
		self.memory_dim = mem_dim
		self.memory_length = mem_size
		# construct memory variables
		self.memory = [tf.Variable(np.zeros(self.memory_dim).astype(np.float32)) for _ in range(mem_size)]

		self.controller = controller
		self.write_vector = [tf.Variable(np.random.rand()) for _ in range(mem_size)]

		# operations
		self.read_op = tf.reduce_sum([a * b for (a,b) in zip(self.write_vector,self.memory)],0)




		# finally initialize all variables
		self.sess.run(tf.global_variables_initializer())
	def read_vector(self):
		self._normalize(self.write_vector)
		return self.sess.run(self.read_op)

	def write(self,erase_v,add_v)
	# normalize a list of tf.Variable and return the new values
	def _normalize(self,vec):
		total = tf.reduce_sum(map(lambda v: tf.abs(v),vec))
		# check if everything is 0
		if total == 0.:
			return sess.run(map(lambda v: v.assign(0.),vec))
		else:
			return sess.run(map(lambda v: v.assign(v/total),vec))

if __name__ == '__main__':
	with tf.Session() as sess:
		ntm = NTM(sess,10,6,None)
		print ntm.read_vector()