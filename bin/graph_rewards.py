#!/usr/bin/env python

import numpy as np
import os
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Graph rewards')
parser.add_argument('name', nargs='?', default='recent', help='Name of directory in ./saves to find rewards')
parser.add_argument('--smooth', type=int, help='Size of window of rewards to average', default=1000)
args = parser.parse_args()

save_path = os.path.join('saves/', args.name)
rewards_path = os.path.join(save_path, 'rewards.txt')

rewards = []
with open(rewards_path, 'r') as f:
    for line in f:
        rewards.append(int(line))

rewards = np.array(rewards)
smooth_n = args.smooth

smoothed_rewards = []
for i in range(len(rewards) - smooth_n):
    smoothed_rewards.append(np.sum(rewards[i:i+smooth_n]) / smooth_n)

print(np.max(smoothed_rewards + [0]))
plt.plot(smoothed_rewards)
plt.savefig(os.path.join(save_path, 'reward-plot.png'))
