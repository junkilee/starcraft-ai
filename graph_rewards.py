import numpy as np
from matplotlib import pyplot as plt

rewards = []
with open('../rewards.txt', 'r') as f:
    for line in f:
        rewards.append(int(line))

rewards = np.array(rewards)
smooth_n = 2500

smoothed_rewards = []
for i in range(len(rewards) - smooth_n):
    smoothed_rewards.append(np.sum(rewards[i:i+smooth_n]) / smooth_n)

print(np.max(smoothed_rewards))
plt.plot(smoothed_rewards)
plt.show()
