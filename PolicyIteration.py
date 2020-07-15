# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:23:15 2020

@author: Grant
"""

import numpy as np
import matplotlib.pyplot as plt
import random

#%% Initialize arrays

grid_size = 4

ncol = grid_size
nrow = grid_size
nS = nrow*ncol
actions = ['left','right', 'up','down']
nA = len(actions)


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

"""
Describe the four types of grids

(S: starting point, safe)
(F: frozen surface, safe)
(H: hole, fall to your doom)
(G: goal)

"""

#generate map
Map = [['S', 'F', 'F', 'F'], ['F', 'H',  'F', 'H'], ['F', 'F',  'F', 'H'], ['H', 'F',  'F', 'G']]
Map = np.asmatrix(Map)

#generate reward per the map
reward_map = np.zeros((len(Map),len(Map)))
for ii in range(len(Map)):
    for jj in range(len(Map)):
        if Map[ii,jj] == 'G':
            reward_map[ii,jj] = 1

#generate probability dictionary.
P = {s: {a: [] for a in range(nA)} for s in range(nS)}

#Define rewards
rewards = [0,1]
STEP_REWARD = 0
LOSE_REWARD = 0
WIN_REWARD = 1

#State matrix. Set of all states on the board
State = np.zeros((grid_size,grid_size))

#pi matrix. i.e. the probability of taking action A under each state S.
pi = {s: [] for s in range(nS)}
#Initialized as equal probability for selecting every point.
for s in range(nS):
    pi[s] = (0.25*np.ones(nA))



#%% Code copied from Frozen Lake OpenAI Gym Environment
# This will help with generating the policy iteration functions

#Goes to the state based on the row and column
def to_s(row, col):
            return row*ncol + col

# Increments the row and column based on the action made
def inc(row, col, a):
    if a == LEFT:
        col = max(col - 1, 0)
    elif a == DOWN:
        row = min(row + 1, nrow - 1)
    elif a == RIGHT:
        col = min(col + 1, ncol - 1)
    elif a == UP:
        row = max(row - 1, 0)
    return (row, col)


def update_probability_matrix(row, col, action):
    newrow, newcol = inc(row, col, action)
    newstate = to_s(newrow, newcol)
    newletter = Map[newrow, newcol]
    if newletter == 'G':
        done = True
        reward = 1
    elif newletter == 'H':
        done = True
        reward = 0
    else: 
        done = False
        reward = 0
    return newstate, reward, done

for row in range(nrow):
    for col in range(ncol):
        s = to_s(row, col)
        for a in range(4):
            li = P[s][a]
            letter = Map[row, col]
            if letter == 'H':
                li.append((1.0, s, 0, True))
            elif letter == 'G':
                li.append((1.0, s, 0, True))
            else:
                li.append((1., *update_probability_matrix(row, col, a)))
                
                

#%%   

#Need to add a tolerance here to check if we have converged!
def policy_evaluation(pi, policy, V, gamma):
    policy_val = np.zeros(nS) 
    
    for s in range(nS):
        pi_mat = np.array(pi[s])
        for a in range(nA):
            for probability, newstate, reward, done in P[s][a]:
                policy_val[s] += pi_mat[a] * probability*(reward + gamma*V[newstate])
    
    return policy_val

#Calculate the action value function for each state and action.
def action_value_function(V, gamma):
    
    action_val = np.zeros((nS,nA))
    for s in range(nS):
        for a in range(nA):
            for probability, newstate, reward, done in P[s][a]:
                action_val[s,a] += probability*(reward + gamma*V[newstate])
    
    return action_val
               

#Calculate the maximum action_value
def policy_update(V, gamma):
    
    action_val = action_value_function(V,gamma)
    
    for s in range(nS):
        pi_mat = pi[s]
        action_val_vect = action_val[s]
        maximumval = np.max(action_val_vect)
        indexval = np.where(action_val_vect == maximumval)
        pi_mat = np.array(np.zeros(nA))
        pi_mat[indexval] = 1
        pi[s] = pi_mat
        
    return pi

#Currently copied and pasted from this link: https://github.com/waqasqammar/MDP-with-Value-Iteration-and-Policy-Iteration/blob/master/MDP_with_PI_VI.ipynb
#I'll want to update this a little bit to make it more of our own
def policy_iteration(max_iter):
    
    # intialize the state-Value function
    V = np.zeros(nS)
    
    # intialize a random policy
    policy = np.random.randint(0, 4, nS)
    policy_prev = np.copy(policy)
    
    for i in range(max_iter):
        
        # evaluate given policy
        V = policy_evaluation(pi, policy, V, gamma)
        
        # improve policy
        policy = policy_update(V, gamma)
        
        # if policy not changed over 10 iterations it converged.
        if i % 10 == 0:
            if (np.all(np.equal(policy, policy_prev))):
                print('policy converged at iteration %d' %(i+1))
                break
            policy_prev = np.copy(policy)
            

    return V, policy  

gamma = 0.5
max_iter = 10000
opt_V2, opt_policy = policy_iteration(max_iter = 10000)
        
        
    
            
            