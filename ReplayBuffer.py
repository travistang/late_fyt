from collections import deque
import random

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def getSequentialBatch(self,seq_len,batch_size):
        if self.num_experiences < seq_len: return None
        if self.num_experiences < batch_size:
            res = [self.buffer[i:i + seq_len] for i in xrange(self.num_experiences - 4)]
            random.shuffle(res)
            return res
        return [self.buffer[i:i + seq_len] for i in sample(xrange(self.num_experiences - 4))]

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0
'''
class History(object):
    def __init__ (self,hist_len):
        self.hist_len = hist_len
        self.history = deque()
        self.num_history = 0
    def add(s_t):
        if self.num_history >= hist_len:
            self.hitory.popleft()
            self.history.append(s_t)
    def get():
        if self.history == 0
'''
