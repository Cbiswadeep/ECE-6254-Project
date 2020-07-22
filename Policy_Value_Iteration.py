# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:23:15 2020

@author: Grant
"""

import numpy as np
import time
import gym

#%% Initialize arguments and environment

#Default environment is such that the frozen lake is slippery (i.e. is_slippery = True)
environment = gym.make('FrozenLake8x8-v0', is_slippery = True)

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
    - Probability = Probability that given your current state and directed action, that you'll end up in the expected new state.
                    Since the default for the environment is that the ice is slippery, there is only a 33% chance
                    that the agent will end up in the correct spot. More on this in #2 below. 
    - nextstate = The next state that we have ended up in, based on the current state and actual action performed.
                  Note that this new state is based off of the actual action, not directed action.
    - Reward = The reward for your agent based on what new state they end up in. 
    - Done = A True/False check to see if your agent has fallen into a hole or hit the goal. 
             Used to signify the end of an episode, and start of a new one.  
   
2) Probabilities of Movement
    - When the agent is in a 'F' or 'S' state, there is a 33% chance that the direction the agent goes 
      is the way you want it to, due to the slippery conditions being True. 
    - For example: if the best action is to move down, then there is a 33% chance that the agent could move
      left and a 33% that the agent could move right. 
    - If the action prescribed is for instance left, there is a zero chance that the agent will go the opposite way, right. 
    - Below is a breakdown of each direction, and the probabilities of going each way. 
        - If directed LEFT: 33% chance of going DOWN, 33% chance of going LEFT, 33% chance of going UP
        - If directed DOWN: 33% chance of going LEFT, 33% chance of going DOWN, 33% chance of going RIGHT
        - If directed RIGHT: 33% chance of going DOWN, 33% chance of going RIGHT, 33% chance of going UP


3) Actions: The actions and subsequent policy that determine your actions are listed below
    - LEFT = 0
    - DOWN = 1
    - RIGHT = 2
    - UP = 3
    - Policy = a vector of size 1 x number_states. The number in the cell corresponds to 
                the greedy policy (direction) for that corresponding state.
               
4) State: The grid of 4x4 or 8x8 is broken up into a 1D vector. The state "number" is
          sequentially added starting top left and ending bottom right. So a 4x4 grid will have
          state numbers from 0 to 15. This nomenclature is used to run a lot of the functions in
          the Gym environment. 

5) Rewards
    - STEP_REWARD = 0
    - WIN_REWARD = 1
    - LOSE_REWARD = 0
    
6) Slippery Conditions
    - As mentioned above, the environment defaults to slippery conditions, signifying that 
      you may not go the direction you want to. 
    - This can be turned off by appending 'is_slippery = False' to the gym.make command in line 15
           

"""

#%% Play Episodes

# Code Pulled from https://github.com/waqasqammar/MDP-with-Value-Iteration-and-Policy-Iteration/blob/master/MDP_with_PI_VI.ipynb

def play_episodes(env, n_episodes, policy, random = False):
    """
    This fucntion plays the given number of episodes given by following a policy or sample randomly from action_space.
    
    Parameters:
        env: openAI GYM object
        n_episodes: number of episodes to run
        policy: Policy to follow while playing an episode
        random: Flag for taking random actions. if True no policy would be followed and action will be taken randomly
        
    Return:
        wins: Total number of wins playing n_episodes
        total_reward: Total reward of n_episodes
        avg_reward: Average reward of n_episodes
    
    """
    # intialize wins and total reward
    wins = 0
    total_reward = 0
    
    # loop over number of episodes to play
    for episode in range(n_episodes):
        
        # flag to check if the game is finished
        terminated = False
        
        # reset the enviorment every time when playing a new episode
        state = env.reset()
        
        while not terminated:
            
            # check if the random flag is not true then follow the given policy other wise take random action
            if random:
                action = env.action_space.sample()
            else:
                action = policy[state]

            # take the next step
            next_state, reward,  terminated, info = env.step(action)
            
            #change the reward structure to negatively reward travel and falling in holes
#            if terminated and reward == 0:
#                reward = -10
#            elif terminated and reward == 1:
#                reward = 10
#            else: 
#                reward = -1
            
            #env.render()
            
            # accumalate total reward
            total_reward += reward
            
            # change the state
            state = next_state
            
            # if game is over with positive reward then add 1.0 in wins
            if terminated and reward == 1.0:
                wins += 1
                
    # calculate average reward
    average_reward = total_reward / n_episodes
    
    return wins, total_reward, average_reward


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
        
        policy_val = np.zeros(nS)  #Initialize the new policy value function (i.e. V(s))
        
        for state, action in enumerate(policy):
            for probablity, next_state, reward, info in env.P[state][action]:
                policy_val[state] += probablity * (reward + (gamma * V[next_state]))
        
        #Calculate maximum difference of state value function at each state
        diff = 0
        for state in range(int(nS)):
            diff = max(diff, np.abs(policy_val[state] - V[state]))
            #print(diff)
        if diff < tol:
            #print('tolerance reached after %d cycles' %ii)
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

def policy_update(env, policy, V, gamma):
    
    """
    function to update a given policy based on given value function.
    """
    
    for state in range(env.nS):
        # for a given state compute state-action value.
        action_val = action_value_function(env, state, V, gamma)
    
        # choose the action which maximizez the state-action value.
        policy[state] =  np.argmax(action_val)
    
    #print(policy)
    return policy



def policy_iteration(env, maxiter):
    
    """
    This function is your high level policy iteration function.
    It goes through the following steps (to iterate over): 
        1) Evaluate your state_value_function, V(s), based on your current policy
        2) Update your policy based on the action_value_function, Q(s,a). 
        3) Repeat until the policy has converged
    
        Outputs: Your final state_value_function, V(s), and optimal policy 'pi'
    """
    
    # intialize the state-Value function
    V = np.zeros(nS)
    
    # intialize a random policy
    policy = np.random.randint(0, 4, nS)
    #print(policy)
    policy_prev = np.copy(policy)
    
    for i in range(maxiter):
        
        # evaluate given policy
        V = policy_evaluation(env, policy, V, gamma, tol, maxiter)
        
        # improve policy
        policy = policy_update(env, policy, V, gamma)
        
        # if policy not changed over 10 iterations it converged.
        if i % 10 == 0:
            if (np.all(np.equal(policy, policy_prev))):
                print('No Changes for 10 iterations. Policy converged at iteration %d' %(i+1))
                break
            policy_prev = np.copy(policy)
            

    return V, policy  
            


#%% Value Iteration Algorithm
"""
The value iteration algorithm uses some of the functions above to help us. 
The algorithm is based off of Sutton and Barto's book, page 83.
"""


def Optimum_V(env, V, maxiter, gamma):
    
    """
    This function finds your optimum state-value function, V(s)
    At each state, calculate the action_value_function, producing a 1x(nA) vector.
    Your new_V(s) is the maximum value over the action_value_function actions 
    
    Output: 
        - The optimum value function, V(s), a 1x(nS) vector
        - The maximum difference over all states between the old V(s) and new V(s)
    """

    new_V = np.zeros(nS)   
    difference = 0
    
    for s in range(nS):
        action_val = action_value_function(env, s, V, gamma)
        new_V[s] = np.max(action_val)
        difference = max(difference, np.abs(V[s] - new_V[s]))

            
    return difference, new_V

def value_iteration(env,maxiter):
    
    """
    Just like policy_iteration, this employs a similar approach.
    Steps (to iterate over):
        1) Find your optimum state_value_function, V(s).
        2) Keep iterating until convergence
        3) Calculate your optimized policy
    
    Outputs: 
        - Your final state_value_function, V(s) 
        - Optimal policy 'pi'
    """
    
    # intialize the state-Value function
    V = np.zeros(nS)
    
    # Iterate over your optimized function, breaking if not changing or difference < tolerance.  
    for i in range(maxiter):
        
        prev_V = np.copy(V)
        
        # evaluate given policy
        difference, V = Optimum_V(env, prev_V, maxiter, gamma)
        
        
        # if State Value function has not changed over 10 iterations, it has converged.
        if i % 10 == 0:
            # if values of 'V' not changing after one iteration
            if (np.all(np.isclose(V, prev_V))):
                print('No Changes for 10 iterations. Value converged at iteration %d' %(i+1))
                break
            
        elif difference < tol:
            print('Tolerance reached. Value converged at iteration %d' %(i+1))
            break
      
    
    # Initialize Optimal Policy
    optimal_policy = np.zeros(nS, dtype = 'int8')
    
    # Update your optimal policy based on optimal value function 'V'
    optimal_policy = policy_update(env, optimal_policy, V, gamma)

    return V, optimal_policy   
        

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

#n_episode = 100
#wins_PI, total_reward_PI, avg_reward_PI = play_episodes(environment, n_episode, opt_policy, random = False)
#print('PI -- Total wins: %d, total reward: %f, Average Reward: %f' %(wins_PI,total_reward_PI,avg_reward_PI))


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


#n_episode = 100
#wins_VI, total_reward_VI, avg_reward_VI = play_episodes(environment, n_episode, opt_policy2, random = False)
#print('VI -- Total wins: %d, total reward: %f, Average Reward: %f' %(wins_VI,total_reward_VI,avg_reward_VI))
