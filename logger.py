import matplotlib.pyplot as plt

class Logger(object):
    def __init__(self,num_channels):
        self.num_channels = num_channels
        self.data = [[] for i in range(num_channels)]
    def _subplot(self,chan_num):
        res = self.num_channels * 100
        res = res + 10 # one column only
        res = res + chan_num
        plt.subplot(res)

    def add_data(self,chan_num,data):
        self._subplot(chan_num)
        self.
