"""
Created on Fri Jul 17 19:10:42 2020
@author: David

This is an algorithm to implement Q learning on a Frozen Lake game

This code is created by using the following Github repo as a template:
    https://github.com/OmarAflak/FrozenLake-QLearning/blob/master/qlearning.py
"""

#initialize
from time import sleep
import numpy as np
import gym

# Environment
env = gym.make('FrozenLake-v0')
inputCount = env.observation_space.n
actionsCount = env.action_space.n

# Init Q-Table
#Set inital values of Q table to random values because we know nothing of the real values
Q = {}
for i in range(inputCount):
    Q[i] = np.random.rand(actionsCount)

"""
The parameters listed are as follows:
  lr = learning rate
  lrMin = minimum learning rate
  lrDecay = rate of decay of learning rate
  gamma = discount factor
  epsilon = variable specifying how often we choose a random action
  epsilonMin = minimum epsilon value
  epsilonDecay = rate of epsilon decay
  episodes = # of training episodes to run

The way this algorithm works is by first initailizing a 'Q-table' which, at each location, stores 'quality' values for each possible action
These quality values are based on anticipated future reward
Initially, the Q table is set randomly and the algorithm updates it each iteration (or episode) based on reward
When the actor goes to make a decision on what action to take, the actor will either pick the action which has the highest q value or pick a random action
The chance that the actor picks a random action rather than an action based on q value is based on the hyperparameter 'epsilon'
This randomness helps the actor explore new paths rather than getting trapped in one path
The epsilon value starts high to encourage exploration but slowly decays each iteration as the specified path should become more clear
"""


# Hyperparameters
lr = 0.33
lrMin = 0.001
lrDecay = 0.9999
gamma = 1.0
epsilon = 1.0
epsilonMin = 0.001
epsilonDecay = 0.97
episodes = 2000

# Training
for i in range(episodes):
    print("Episode {}/{}".format(i + 1, episodes))
    s = env.reset()
    done = False

    while not done:
        if np.random.random() < epsilon:
            a = np.random.randint(0, actionsCount)
        else:
            a = np.argmax(Q[s])

        newS, r, done, _ = env.step(a)
        Q[s][a] = Q[s][a] + lr * (r + gamma * np.max(Q[newS]) - Q[s][a])
        s = newS

        if lr > lrMin:
            lr *= lrDecay

        if not r==0 and epsilon > epsilonMin:
            epsilon *= epsilonDecay


print("")
print("Learning Rate :", lr)
print("Epsilon :", epsilon)

# Testing
print("\nPlay Game on 100 episodes...")

avg_r = 0
for i in range(100):
    s = env.reset()
    done = False

    while not done:
        a = np.argmax(Q[s])
        newS, r, done, _ = env.step(a)
        s = newS

    avg_r += r/100.

print("Average reward on 100 episodes :", avg_r)
