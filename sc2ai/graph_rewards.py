import numpy as np
from matplotlib import pyplot as plt

rewards = []
with open('rewards.txt', 'r') as f:
    for line in f:
        rewards.append(int(line))

smoothed_rewards = []
for i in range(len(rewards) - 100):
    smoothed_rewards.append(np.sum(rewards[i:i+100]) / 100)

plt.plot(smoothed_rewards)
plt.show()
