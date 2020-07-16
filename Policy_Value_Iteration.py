# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:23:15 2020

@author: Grant
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import gym

#%% Initialize arguments and environment

environment = gym.make('FrozenLake-v0')

nA = environment.nA
nS = environment.nS
nrow = environment.nrow
ncol = environment.ncol

#Setting up parameters to use later
gamma = 0.999
tol = 1e-6
maxiter = 10000

#%% FrozenLake Environment Important Notes: 
    
"""
FrozenLake source code: https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In order to run this software, gym must be installed via Pip or Conda. 
Feel free to ask if you're having trouble getting it installed. Some documentation on how to install: 
    - https://anaconda.org/conda-forge/gym
    - http://gym.openai.com/docs/
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are some important functions,dictionaries,variables that I am using from the particular environment
in Gym that are helpful to know:

1) P[s][a] 
    - Def: A dictionary that for all states and actions, you have FOUR parameters:
           (probability, nextstate, reward, done). See below for explanation of each.
    - Probability = Probability that given your current state and action, that you'll end up in the new state.
                    Based on how the next states are generated, this probability is always 1. 
                    For example: if we are in state (1,1) and move to the right, there is a 100% chance 
                    we'll end up at our new state of (1,2). This is also called the nextstate. 
    - nextstate = The next state that we have ended up in, based on the current state and action performed.
                  The way that the source code is written, this next state is always true to the prior state
                  and action. See above example
    - Reward = The reward for your agent based on what new state they end up in. 
    - Done = A True/False check to see if your agent has fallen into a hole or hit the goal. 
             Used to signify the end of an episode, and start of a new one.  
   

2) Actions: The actions and subsequent policy that determine your actions are listed below
    - LEFT = 0
    - DOWN = 1
    - RIGHT = 2
    - UP = 3
    - Policy = a vector of size 1 x number_states. The number in the cell corresponds to 
                the greedy policy (direction) for that corresponding state.
               
3) State: The grid of 4x4 or 8x8 is broken up into a 1D vector. The state "number" is
          sequentially added starting top left and ending bottom right. So a 4x4 grid will have
          state numbers from 0 to 15. This nomenclature is used to run a lot of the functions in
          the Gym environment. 

3) Rewards
    - STEP_REWARD = 0
    - WIN_REWARD = 1
    - LOSE_REWARD = 0
           

"""


#%% Policy Iteration Algorithm

"""
Performing Policy Iteration per the pseudo code in Sutton & Barto's RL book, page 80.
Also getting inspiration from two Github repo's (Using their functions as templates):
   1) https://github.com/waqasqammar/MDP-with-Value-Iteration-and-Policy-Iteration/blob/master/MDP_with_PI_VI.ipynb
   2) https://github.com/sanuj/frozen-lake/blob/master/PI.ipynb
"""

def policy_evaluation(env, policy, V, gamma, tol, maxiter):
    
    """ 
    Evaluate your current policy 'pi(s)' to find the state value function V_{pi}(s).
    At each state, you sum up the state_value_Function for all next possible 
    states (determined by your available actions). Compare the difference between the new value
    and your old V(s). If all differences over the states are less 
    than some tolerance, then we move forward. If not, we continue iterating until it is.
    
    Output: Optimized state value function for your policy (i.e. v_{pi}(s)). Size is 1 x (nS) vector
    """
    
    for ii in range(maxiter): 
        
        difference = 0
        policy_val = np.zeros(nS)  #Initialize the new policy value function (i.e. V(s))
        
        for state, action in enumerate(policy):
            for probablity, next_state, reward, info in env.P[state][action]:
                
                policy_val[state] += probablity * (reward + (gamma * V[next_state]))
                difference = max(difference,np.abs(policy_val[state] - V[state]))
                
        if difference < tol:
            break
        else:
            V = policy_val
        
    return policy_val 
               


def action_value_function(env, state, V , gamma):
    
    """
    Function to  calculate action-value function at some input state, Q(s,a). 
    
    Output: The action_value_function, a 1x(nA) vector, that is unique for your input state.
    """
    
    # initialize vector of action values
    action_val = np.zeros(nA)
    
    # loop over the actions we can take in an enviorment 
    for action in range(nA):
        # loop over the P_sa distribution
        for probablity, next_state, reward, info in env.P[state][action]:
             #if we are in state s and take action a. then sum over all the possible states we can land into.
            action_val[action] += probablity * (reward + (gamma * V[next_state]))
            
    return action_val

def policy_update(env, policy, V):
    
    """
    function to update a given policy based on given value function.
    """
    
    for state in range(env.nS):
        # for a given state compute state-action value.
        action_val = action_value_function(env, state, V, gamma)
    
        # choose the action which maximizez the state-action value.
        policy[state] =  np.argmax(action_val)
    
    print(policy)
    return policy

#Currently copied and pasted from this link: https://github.com/waqasqammar/MDP-with-Value-Iteration-and-Policy-Iteration/blob/master/MDP_with_PI_VI.ipynb
#I'll want to update this a little bit to make it more of our own
def policy_iteration(env, maxiter):
    
    """
    Steps (to iterate over): 
        1) Evaluate your state_value_function, V(s), based on your current policy
        2) Update your policy based on the action_value_function, Q(s,a). 
        3) Repeat until convergence
    
        Outputs: Your final state_value_function, V(s), and optimal policy 'pi'
    """
    
    # intialize the state-Value function
    V = np.zeros(nS)
    
    # intialize a random policy
    policy = np.random.randint(0, 4, nS)
    print(policy)
    policy_prev = np.copy(policy)
    
    for i in range(maxiter):
        
        # evaluate given policy
        V = policy_evaluation(env, policy, V, gamma, tol, maxiter)
        
        # improve policy
        policy = policy_update(env, policy, V)
        
        # if policy not changed over 10 iterations it converged.
        if i % 10 == 0:
            if (np.all(np.equal(policy, policy_prev))):
                print('policy converged at iteration %d' %(i+1))
                break
            policy_prev = np.copy(policy)
            

    return V, policy  
            


#%% Value Iteration Algorithm
"""
The value iteration algorithm uses some of the functions above to help us. 
The algorithm is based off of Sutton and Barto's book, page 83.
"""


def Optimum_V(env, V, policy, maxiter, gamma, tol):
    
    """
    This function finds your optimum state-value function, V(s)
    At each state, calculate the action_value_function, producing a 1x(nA) vector.
    Your new_V(s) is the maximum value over the action_value_function actions 
    Take the difference with our old V. If its less than some tolerance, then we move forward. 
    If not, we continue iterating until it is.
    """

    new_V = np.zeros(nS)
    for ii in range(maxiter):
        difference = 0
        for s in range(nS):
            action_val = action_value_function(env, s, V, gamma)
            new_V[s] = np.max(action_val)
            difference = max(difference, np.abs(V[s] - new_V[s]))
        if difference < tol:
            break
        else: 
            V = new_V
            
    return new_V

def value_iteration(env,maxiter):
    
    """
    Just like policy_iteration, this employs a similar approach.
    Steps (to iterate over):
        1) Find your optimum state_value_function, V(s). 
        2) Determine the new policy based on this V.
        3) Rinse repeat.
    
    Outputs: Your final state_value_function, V(s), and optimal policy 'pi'
    """
    
    # intialize the state-Value function
    V = np.zeros(nS)
    
    # intialize a random policy
    policy = np.random.randint(0, 4, nS)
    print(policy)
    policy_prev = np.copy(policy)
    
    for i in range(maxiter):
        
        # evaluate given policy
        V = Optimum_V(env, V, policy, maxiter, gamma, tol)
        
        # improve policy
        policy = policy_update(env, policy, V)
        
        # if policy not changed over 10 iterations it converged.
        if i % 10 == 0:
            if (np.all(np.equal(policy, policy_prev))):
                print('policy converged at iteration %d' %(i+1))
                break
            policy_prev = np.copy(policy)
            

    return V, policy   
        

#%% Run Policy Iteration        
        
tic = time.time()
opt_V, opt_policy = policy_iteration(environment, maxiter)
toc = time.time()
elapsed_time = (toc - tic) * 1000
print (f"Time to converge: {elapsed_time: 0.3} ms")
print('Optimal Value function: ')
print(opt_V.reshape((nrow, ncol)))
print('Final Policy: ')
print(opt_policy.reshape(nrow,ncol))


#%% Run Value Iteration

tic = time.time()
opt_V2, opt_policy2 = value_iteration(environment, maxiter)
toc = time.time()
elapsed_time = (toc - tic) * 1000
print (f"Time to converge: {elapsed_time: 0.3} ms")
print('Optimal Value function: ')
print(opt_V2.reshape((nrow, ncol)))
print('Final Policy: ')
print(opt_policy2.reshape(nrow,ncol))
