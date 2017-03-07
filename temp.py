import numpy as np
import matplotlib.pyplot as plt

plt.ion()
colors = [np.random.uniform(0,1,(1,3)) for i in range(10)]
for i in range(10):
    y = np.random.random()
    plt.scatter(i, y,c = colors[i])

while True:
    plt.pause(0.05)
