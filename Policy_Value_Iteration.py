# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:23:15 2020

@author: Grant
"""

import numpy as np
import time
import gym
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import seaborn
import os
from os import startfile

#%% Filepaths for saving videos
fn_PIvid = 'PI_Heatmap.mp4'     #filename for PI heatmap video
fn_PIplot = 'PI_Reward.png'     #filename for PI reward graph
fn_VIvid = 'VI_Heatmap.mp4'     #filename for VI heatmap video
fn_VIplot = 'VI_Reward.png'     #filename for VI reward graph
cdir = os.path.abspath(os.getcwd()) #current directory (directory files will be saved to)
viddir_PI = os.path.join(cdir,fn_PIvid)     #directory PI video will be saved to
figdir_PI = os.path.join(cdir,fn_PIplot)    #directory PI graph will be saved to
viddir_VI = os.path.join(cdir,fn_VIvid)     #directory PI video will be saved to
figdir_VI = os.path.join(cdir,fn_VIplot)    #directory PI graph will be saved to

#%% Initialize arguments and environment
#Default environment is such that the frozen lake is slippery (i.e. is_slippery = True)
environment = gym.make('FrozenLake8x8-v0', is_slippery=True)

#Default environment is such that the frozen lake is slippery (i.e. is_slippery = True)
#custom_map = ['SFFF','FHFH','FFFH','HFFG']  #standard 4x4 map
custom_map = ['SFFFFFFF', 'FFFFFFFF', 'FFFHFFFF', 'FFFFFHFF', 'FFFHFFFF', 'FHHFFFHF', 'FHFFHFHF', 'FFFHFFFG'] #standard 8x8 map
#environment = gym.make('FrozenLake-v0', is_slippery=True, desc=custom_map)

print("")
print('Lake Visualization:')
print('S=Start, F=Frozen, H=Hole, G=Goal')
environment.render()

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
    
        Outputs: 
            - Your final state_value_function, V(s) 
            - Optimal policy 'pi'
            - Average reward based on 100 episodes
            - List of all value functions and policies for all iterations (V_hm, P_hm)
    """
    
    # intialize the state-Value function
    V = np.zeros(nS)
    V_hm = np.copy(V)
    V_hm.resize((1,V_hm.size))
    V_hm = V_hm.tolist()
    
    # intialize a random policy
    policy = np.random.randint(0, 4, nS)
    #print(policy)
    policy_prev = np.copy(policy)
    P_hm = np.copy(policy)
    P_hm.resize((1,P_hm.size))
    P_hm = P_hm.tolist()
    
    #Initialize an average reward vector
    avg_r_PI_mat = []
    n_episode = 100
    
    for i in range(maxiter):
        
        # evaluate given policy
        V = policy_evaluation(env, policy, V, gamma, tol, maxiter)
        # save value function to list for animation
        V_tmp = np.copy(V)
        V_tmp = V_tmp.tolist()
        V_hm.append(V_tmp)
        
        # improve policy
        policy = policy_update(env, policy, V, gamma)
        # save policy to list for animation
        P_tmp = np.copy(policy)
        P_tmp = P_tmp.tolist()
        P_hm.append(P_tmp)
        
        #Play episodes based on the current policy.
        wins_PI, total_reward_PI, avg_reward_PI = play_episodes(env, n_episode, policy, random = False)
        avg_r_PI_mat.append(avg_reward_PI)
        
        
        # if policy not changed over 10 iterations it converged.
        if i % 10 == 0:
            if (np.all(np.equal(policy, policy_prev))):
                print("")
                print('No Changes for 10 iterations. Policy converged at iteration %d' %(i+1))
                break
            policy_prev = np.copy(policy)
            

    return V, policy, avg_r_PI_mat, V_hm, P_hm
            


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
        - Average reward vector (see note below)
        - List of all value functions for all iterations
        
    
    NOTE: In order to produce the graph showing average reward over each
          iteration, the policy was calculated at each iteration. This is not
          normally done for Value Iteration. This will slow down the computation
          time for Value iteration. To return to traditional value iteration, 
          comment out the respective lines and remove the appropriate output
    """
    
    # intialize the state-Value function
    V = np.zeros(nS)
    V_hm = np.copy(V)
    V_hm.resize((1,V_hm.size))
    V_hm = V_hm.tolist()
    
    # intialize a random policy. Comment out for traditional Value_Iteration
    policy = np.random.randint(0, 4, nS)
    avg_r_VI_mat = []
    n_episode = 100
    
    
    # Iterate over your optimized function, breaking if not changing or difference < tolerance.  
    for i in range(maxiter):
        
        prev_V = np.copy(V)
        
        # evaluate given policy
        difference, V = Optimum_V(env, prev_V, maxiter, gamma)
        
        # improve policy. Comment out to return to traditional Value Iteration
        policy = policy_update(env, policy, V, gamma)
        
        #Play episodes based on the current policy. Comment out to return to traditional Value Iteration
        wins_VI, total_reward_VI, avg_reward_VI = play_episodes(env, n_episode, policy, random = False)
        avg_r_VI_mat.append(avg_reward_VI)
    
        # save value function to list for animation
        V_tmp = np.copy(V)
        V_tmp = V_tmp.tolist()
        V_hm.append(V_tmp)
        
        # if State Value function has not changed over 10 iterations, it has converged.
        if i % 10 == 0:
            # if values of 'V' not changing after one iteration
            if (np.all(np.isclose(V, prev_V))):
                print("")
                print('No Changes for 10 iterations. Value converged at iteration %d' %(i+1))
                break
            
        elif difference < tol:
            print('Tolerance reached. Value converged at iteration %d' %(i+1))
            break
      
    
    # Initialize Optimal Policy
    optimal_policy = np.zeros(nS, dtype = 'int8')
    
    # Update your optimal policy based on optimal value function 'V'
    optimal_policy = policy_update(env, optimal_policy, V, gamma)

    return V, optimal_policy, avg_r_VI_mat, V_hm

#%% Run Policy Iteration        
        
tic = time.time()
opt_V, opt_policy, avg_r_PI_mat, V_PI, P_PI = policy_iteration(environment, maxiter)
toc = time.time()
elapsed_time = (toc - tic) * 1000
print (f"Time to converge: {elapsed_time: 0.3} ms")
print('Optimal Value function: ')
print(opt_V.reshape((nrow, ncol)))
print('Final Policy: ')
print(opt_policy.reshape(nrow,ncol))

#n_episode = 1000
#wins_PI, total_reward_PI, avg_reward_PI = play_episodes(environment, n_episode, opt_policy, random = False)
#print('PI -- Total wins: %d, total reward: %f, Average Reward: %f' %(wins_PI,total_reward_PI,avg_reward_PI))


#%% Run Value Iteration

tic = time.time()
opt_V2, opt_policy2, avg_r_VI_mat, V_VI = value_iteration(environment, maxiter)
toc = time.time()
elapsed_time = (toc - tic) * 1000
print (f"Time to converge: {elapsed_time: 0.3} ms")
print('Optimal Value function: ')
print(opt_V2.reshape((nrow, ncol)))
print('Final Policy: ')
print(opt_policy2.reshape(nrow,ncol))

#n_episode = 1000
#wins_VI, total_reward_VI, avg_reward_VI = play_episodes(environment, n_episode, opt_policy2, random = False)
#print('VI -- Total wins: %d, total reward: %f, Average Reward: %f' %(wins_VI,total_reward_VI,avg_reward_VI))


#%% Plot average reward
plt.figure(1)
plt.plot(range(len(avg_r_PI_mat)), avg_r_PI_mat)
plt.xlabel('Iteration')
plt.ylabel('Average Reward')
plt.title('Policy Iteration - Average Reward over 100 Episodes')
plt.grid()

plt.figure(2)
plt.plot(range(len(avg_r_VI_mat)), avg_r_VI_mat)
plt.xlabel('Iteration')
plt.ylabel('Average Reward')
plt.title('Value Iteration - Average Reward over 100 Episodes')
plt.grid()

#%% Functions to Create Animations
def animatePIheatmap(j):
    plt.cla()
    V_PI_reshaped = np.reshape(np.array(V_PI[j]),(nrow,ncol))
    p = seaborn.heatmap(V_PI_reshaped, cmap=cmap, vmin=0, vmax=1, cbar=False,
                square=True, xticklabels=ncol+1, yticklabels=nrow+1,
                linewidths=.5, ax=ax3, annot=True, fmt=".3f")
    for i in range(nrow):
        for ii in range(ncol):
            plt.text(i+0.4,ii+0.25,custom_map[ii][i],fontsize=14)
    
    for k in range(len(P_PI[j])):   #len(P_PI[j]) should equal nrow*ncol
        xk = (k % ncol) + 0.4
        yk = (k // nrow) + 0.8
        arrow = P_PI[j][k]
        if arrow == 0:
            plt.text(xk, yk, u'\u2190', fontsize=14)
        elif arrow == 1:
            plt.text(xk, yk, u'\u2193', fontsize=14)
        elif arrow == 2:
            plt.text(xk, yk, u'\u2192', fontsize=14)
        else:
            plt.text(xk, yk, u'\u2191', fontsize=14)
    
    plt.title('Iteration %i' %j)
    
    return p


def animateVIheatmap(j):
    plt.cla()
    V_VI_reshaped = np.reshape(np.array(V_VI_small[j]),(nrow,ncol))
    p = seaborn.heatmap(V_VI_reshaped, cmap=cmap, vmin=0, vmax=1, cbar=False,
                square=True, xticklabels=ncol+1, yticklabels=nrow+1,
                linewidths=.5, ax=ax4, annot=True, fmt=".3f")
    for i in range(nrow):
        for ii in range(ncol):
            plt.text(i+0.4,ii+0.25,custom_map[ii][i],fontsize=14)
    
    #if j == len(VI_it)-1:
    for k in range(len(P_VI_small[j])):   #len(P_VI[j]) should equal nrow*ncol
        xk = (k % ncol) + 0.4
        yk = (k // nrow) + 0.8
        arrow = P_VI_small[j][k]
        if arrow == 0:
            plt.text(xk, yk, u'\u2190', fontsize=14)
        elif arrow == 1:
            plt.text(xk, yk, u'\u2193', fontsize=14)
        elif arrow == 2:
            plt.text(xk, yk, u'\u2192', fontsize=14)
        else:
            plt.text(xk, yk, u'\u2191', fontsize=14)
    
    plt.title('Iteration %i' %int(VI_it[j]))
    
    return p
    
#%% Create animations
print("")
print('Generating Visuals...')

#Create evolving Value Function heatmap for Policy Iteration
plt.figure(3)   #new figure
fig3, ax3 = plt.subplots(figsize=(11, 9))  #needs to be a subplot to call the axis
#cmap = seaborn.diverging_palette(10, 220, sep=80, as_cmap=True) #define diverging colormap
cmap = seaborn.light_palette((210, 90, 60), input="husl", as_cmap=True) #define colormap (not diverging)
V_PI_reshaped = np.reshape(np.array(V_PI[0]),(nrow,ncol))    #reshape values from first iteration of V into array the size of the lake
#draw the heatmap
p = seaborn.heatmap(V_PI_reshaped, cmap=cmap, vmin=0, vmax=1,
            square=True, xticklabels=ncol+1, yticklabels=nrow+1,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax3, annot=True, fmt=".3f")
#Add labels to each point on lake (start, hole, frozen, goal)
for i in range(nrow):
    for ii in range(ncol):
        plt.text(i+0.4,ii+0.2,custom_map[ii][i],fontsize=14)
    
#Reset the counter variables for the animation function.
# for some reason if j isn't reset, it can get buggy       
j=0
k=0
#Create and save Visual #3 (Heatmap video)
ani = animation.FuncAnimation(fig3, animatePIheatmap, interval=1, save_count=len(V_PI))

plt.rcParams['animation.ffmpeg_path'] ='C:\\Program Files\\Python\\ffmpeg-20200713-7772666-win64-static\\bin\\ffmpeg.exe'
FFwriter=animation.FFMpegWriter(fps=4, extra_args=['-vcodec', 'libx264'])
ani.save(fn_PIvid, writer=FFwriter)

plt.show()

#Create evolving Value Function heatmap for Value Iteration
#since VI has a lot more iterations, this will create an animation where we only
#   show ever VI_it iteration (if VI_it_gap=5, we will show iterations 0, 5, 10, ..., n)
VI_it_gap = 5

V_VI_small = [V_VI[0].copy()]
VI_it = [0]
P_VI_tmp = policy_update(environment, np.zeros(nS, dtype = 'int8'), np.array(V_VI[0]), gamma)
P_VI_small = [P_VI_tmp.tolist()]
for i in range(1,len(V_VI)):
    if i % VI_it_gap == 0 or i == len(V_VI)-1:
        V_VI_small.append(V_VI[i])
        VI_it.append(i)
        P_VI_tmp = policy_update(environment, np.zeros(nS,dtype = 'int8'),np.array(V_VI[i]),gamma)
        P_VI_small.append(P_VI_tmp.tolist())


plt.figure(4)   #new figure
fig4, ax4 = plt.subplots(figsize=(11, 9))  #needs to be a subplot to call the axis
#cmap = seaborn.diverging_palette(10, 220, sep=80, as_cmap=True) #define diverging colormap
cmap = seaborn.light_palette((210, 90, 60), input="husl", as_cmap=True) #define colormap (not diverging)
V_VI_reshaped = np.reshape(np.array(V_VI_small[0]),(nrow,ncol))    #reshape values from first iteration of V into array the size of the lake
#draw the heatmap
p = seaborn.heatmap(V_VI_reshaped, cmap=cmap, vmin=0, vmax=1,
            square=True, xticklabels=ncol+1, yticklabels=nrow+1,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax4, annot=True, fmt=".3f")
#Add labels to each point on lake (start, hole, frozen, goal)
for i in range(nrow):
    for ii in range(ncol):
        plt.text(i+0.4,ii+0.2,custom_map[ii][i],fontsize=14)
    
#Reset the counter variables for the animation function.
# for some reason if j isn't reset, it can get buggy       
j=0
k=0
#Create and save Visual #3 (Heatmap video)
ani = animation.FuncAnimation(fig4, animateVIheatmap, interval=1, save_count=len(V_VI_small))

plt.rcParams['animation.ffmpeg_path'] ='C:\\Program Files\\Python\\ffmpeg-20200713-7772666-win64-static\\bin\\ffmpeg.exe'
FFwriter=animation.FFMpegWriter(fps=16, extra_args=['-vcodec', 'libx264'])
ani.save(fn_VIvid, writer=FFwriter)

plt.show()

print('Done')

#Automatically open visualization files
# startfile(viddir_PI)
# startfile(figdir_PI)
# startfile(viddir_VI)
# startfile(figdir_VI)