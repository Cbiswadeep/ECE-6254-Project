"""
Created on Fri Jul 17 19:10:42 2020
@author: David
Edited on Jul 21, 2020 by Jason

This is an algorithm to implement Q learning on a Frozen Lake game

This code is created by using the following Github repo as a template:
    https://github.com/OmarAflak/FrozenLake-QLearning/blob/master/qlearning.py
"""
#%% INITIALIZATION AND FILE DIRECTORIES

#initialize
from time import sleep
import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import seaborn
import os
from os import startfile

fn_path = 'QL_Iterations.mp4'    #filename for path video
fn_epsilon = 'QL_Eps.png'       #filename for epsilon vs iteration graph
fn_heatmap = 'QL_Heatmap.mp4'   #filename for heatmap video
cdir = os.path.abspath(os.getcwd()) #current directory (directory files will be saved to)
viddir_path = os.path.join(cdir,fn_path) #directory path video will be saved to
figdir_eps = os.path.join(cdir,fn_epsilon)  #directory graph image will be saved to
viddir_hm = os.path.join(cdir,fn_heatmap)   #directory heatmap video will be saved to

#%% GENERATE ENVIRONMENT
# Environment
custom_map = ['SFFFFFFF', 'FFFFFFFF', 'FFFHFFFF', 'FFFFFHFF', 'FFFHFFFF', 'FHHFFFHF', 'FHFFHFHF', 'FFFHFFFG']  # added by Jason - standard 4x4 map (uncomment for custom map)
##env = gym.make('FrozenLake-v0', is_slippery=False, desc=custom_map)
env = gym.make('FrozenLake8x8-v0')
print("")
print('Lake Visualization:')
print('S=Start, F=Frozen, H=Hole, G=Goal')
env.render()

#Create numerical representation of frozen lake (as an array)
#-1 = start, 0 = hole, 1 = frozen, 2 = end
nrow = env.nrow
ncol = env.ncol
lake = [[None]*ncol for _ in range(nrow)]
for i in range(0,len(custom_map)):
    tmpstr = custom_map[i]
    
    for j in range(0,len(tmpstr)):
        if tmpstr[j] == 'S':
            lake[i][j] = -1
        elif tmpstr[j] == 'H':
            lake[i][j] = 0
        elif tmpstr[j] == 'F':
            lake[i][j] = 1
        elif tmpstr[j] == 'G':
            lake[i][j] = 2
lake = np.array(lake)   

inputCount = env.observation_space.n
actionsCount = env.action_space.n
      
#%% DEFINE FUNCTIONS FOR VISUALIZATION

vizit = 250     #number of iterations between path visualizations
hmit = 5    #numer of iterations between heatmap visualizations

def movefun(s0,s1,moves):
    if s1-s0 == -1:         #robot moved left = 0
        moves.append(0)
    elif s1-s0 == ncol:     #robot moved down = 1
        moves.append(1)
    elif s1-s0 == 1:        #robot moved right = 2
        moves.append(2)
    elif s1-s0 == -ncol:    #robut moved up = 3
        moves.append(3)
    else:                   #robot didn't move = 4
        moves.append(4)
    
    return moves

def robotloc(moves,x0,y0):
    x = [None]*(len(moves))
    y = [None]*(len(moves))
    
    i = 0
    for m in moves: 
        if m == 0:
            x[i] = x[i-1]-1
            y[i] = y[i-1]
        elif m == 1:
            x[i] = x[i-1]
            y[i] = y[i-1]-1
        elif m == 2:
            x[i] = x[i-1]+1
            y[i] = y[i-1]
        elif m == 3:
            x[i] = x[i-1]
            y[i] = y[i-1]+1
        elif m == 4:
            x[i] = x[i-1]
            y[i] = y[i-1]
        else:
            x[i] = x0
            y[i] = y0
                
        i+=1
    return x, y

def animatepath(j):
    if j == 0:
        line.set_xdata(x[j])
        line.set_ydata(y[j])
    else:
        k=j
        while k > 0:
            if moves[k] == -1:
                break
            k = k-1
        line.set_xdata(x[k:j+1])
        line.set_ydata(y[k:j+1])
        point.set_xdata(x[j])
        point.set_ydata(y[j])
        plt.title('Iteration %i' %itnum[j])

    return line,

def animateheatmap(j):
    plt.cla()
    Val_reshaped = np.reshape(Qval[j,:],(nrow,ncol))
    p = seaborn.heatmap(Val_reshaped, cmap=cmap, vmin=0, vmax=vmax, cbar=False,
                square=True, xticklabels=ncol+1, yticklabels=nrow+1,
                linewidths=.5, ax=ax3, annot=True, fmt="f")
    for i in range(len(lake)):
        for ii in range(len(lake[0])):
            plt.text(i+0.4,ii+0.25,custom_map[ii][i],fontsize=14)
    
    for k in range(len(Qdir[j])):
        xk = (k % ncol) + 0.4
        yk = (k // nrow) + 0.8
        arrow = Qdir[j,k]
        if arrow == 0:
            plt.text(xk, yk, u'\u2190', fontsize=14)
        elif arrow == 1:
            plt.text(xk, yk, u'\u2193', fontsize=14)
        elif arrow == 2:
            plt.text(xk, yk, u'\u2192', fontsize=14)
        else:
            plt.text(xk, yk, u'\u2191', fontsize=14)
    
        plt.title('Iteration %i' %Qit[j])
    
    return p
######################## END ADDED BY JASON ###################################

#%% TRAIN ROBOT
# Initialize Q-Table
inputCount = env.observation_space.n
actionsCount = env.action_space.n
#Set inital values of Q table to zero
Q = {}
for i in range(inputCount):
    Q[i] = np.zeros(actionsCount)

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
lrDecay = 0.99999
gamma = 0.9
epsilon = 1.0
epsilonMin = 0.001
epsilonDecay = 0.97
episodes = 10000
rewardWin = 100
rewardLose = -100
rewardMove = -1

# Variables needed to create visualizations
epsplot = [epsilon]
Qval = np.zeros((int(np.ceil(episodes/hmit)+1),int(nrow*ncol)))
Qdir = np.zeros((int(np.ceil(episodes/hmit)+1),int(nrow*ncol)))
Qit = np.zeros(int(np.ceil(episodes/hmit)+1))


# Training
for i in range(episodes):
    #print("Episode {}/{}".format(i + 1, episodes))
    s = env.reset()
    done = False

    # update list of moves to say it reset - for path movie
    if i == 0 or (i+1) % vizit == 0 or i == episodes-1:
        if i == 0:
            moves = [-1]
            itnum = [1]
        else:
            moves.append(-1)
            itnum.append(i+1)

    # Iterate the path
    while not done:
        #determine if we want to explore or base our action on the Q table
        if np.random.random() < epsilon:
            a = np.random.randint(0, actionsCount)
        else:
            a = np.argmax(Q[s])
        
        #evaluate the result of the action taken
        newS, r, done, _ = env.step(a)
        
        #Manually change the reward structure to negatively reward travel and falling in holes and positvely reward reaching the goal
        if done and r==0:
            r = rewardLose
        elif done and r==1:
            r = rewardWin
        else:
            r = rewardMove
        
        #Update the Q table with the Bellman equation
        Q[s][a] = Q[s][a] + lr * (r + gamma * np.max(Q[newS]) - Q[s][a])
        
        #Update moves array with the last move - for path movie
        if i == 0 or (i+1) % vizit == 0 or i == episodes-1:
            moves = movefun(s,newS,moves)
            itnum.append(i+1)

        # Update s, learning rate, and epsilon
        s = newS
        
        #decay the learning rate
        if lr > lrMin:
            lr *= lrDecay

        #decrease exploration rate if we reach the goal
        if r==rewardWin and epsilon > epsilonMin:
            epsilon *= epsilonDecay
    
    #store epsilon value for epsilon vs iteration plot
    epsplot.append(epsilon)
    
    #store Q matrix values for heatmap plot
    if i==0:
        Qit[i] = i+1
        for tile in range(nrow*ncol):
            Qval[i,tile] = max(Q[tile])
            Qdir[i,tile] = np.argmax(Q[tile])
    elif (i+1) % hmit == 0:
        Qit[int((i+1)/hmit)] = i+1
        for tile in range(nrow*ncol):
            Qval[int((i+1)/hmit),tile] = max(Q[tile])
            Qdir[int((i+1)/hmit),tile] = np.argmax(Q[tile])
    elif i == episodes-1:
        Qit[-1] = i+1
        for tile in range(nrow*ncol):
            Qval[-1,tile] = max(Q[tile])
            Qdir[-1,tile] = np.argmax(Q[tile])
                

print("")
print("Learning Rate :", lr)
print("Epsilon :", epsilon)

#%% TEST SOLUTION ON FROZEN LAKE
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

print("Number of successes out of 100 episodes :", np.round(avg_r*100))


#%% CREATE GRAPHICS AND MOVIES
# 1) Video of the path that is generated on selected iterations
# 2) Graph of Epsilon vs Iteration
# 3) Video of the evolution of the Q-Dictionary

print("")
print('Generating Visuals...')

# 1 - Path Video
#create figure
plt.figure(1)
fig1, ax1 = plt.subplots()
#create lake background
plt.xlim(0,ncol)
plt.ylim(0,nrow)
plt.axis('square')
colormap = ListedColormap(['g','b','c','y'])
c = ax1.pcolor(np.linspace(0,ncol,ncol+1),np.linspace(0,nrow,nrow+1),list(lake[::-1]),cmap=colormap,alpha=0.4)       

#place points on lake to generate variables to call in animation
#   each data set (the line and the final point) have a separate name
x, y = robotloc(moves,0.5,nrow-.5)
line, = ax1.plot(x[0],y[0],'ko-')
point, = ax1.plot(x[0],y[0],'r*',markersize=15)
plt.title('Iteration 0')

#Reset the counter variables for the animation function.
# for some reason if j isn't reset, it can get buggy
j=0
k=0
#Create and save Visual #1 (Path video)
ani = animation.FuncAnimation(fig1, animatepath, interval=1, save_count=len(x))

plt.rcParams['animation.ffmpeg_path'] ='C:\\Program Files\\Python\\ffmpeg-20200713-7772666-win64-static\\bin\\ffmpeg.exe'
FFwriter=animation.FFMpegWriter(fps=10, extra_args=['-vcodec', 'libx264'])
ani.save(fn_path, writer=FFwriter)

plt.show()

############################################################################

#Create plot of epsilon vs iteration
fig2 = plt.figure(2)
plt.plot(np.linspace(1,len(epsplot),len(epsplot)),epsplot)
plt.plot([1,len(epsplot)],[epsilonMin,epsilonMin],'r--')
plt.title('Epsilon Value')
plt.xlabel('Iteration')
plt.ylabel('Epsilon')
plt.legend(['Epsilon','Threshold'])

plt.savefig(figdir_eps)

plt.show()

#############################################################################
#Create evolving Q-dictionary heatmap
plt.figure(3)   #new figure
fig3, ax3 = plt.subplots()  #needs to be a subplot to call the axis

vmax = np.ceil(np.max(Qval)*100)/100    #max value in Qval for upper limit of heatmap color scale
cmap = seaborn.light_palette((210, 90, 60), input="husl", as_cmap=True) #define colormap
Val_reshaped = np.reshape(Qval[0,:],(nrow,ncol))    #reshape values from Qval into array the size of the lake
#draw the heatmap
p = seaborn.heatmap(Val_reshaped, cmap=cmap, vmin=0, vmax=vmax,
            square=True, xticklabels=ncol+1, yticklabels=nrow+1,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax3, annot=True, fmt="f")
#Add labels to each point on lake (start, hole, frozen, goal)
for i in range(len(lake)):
    for ii in range(len(lake[0])):
        plt.text(i+0.4,ii+0.2,custom_map[ii][i],fontsize=14)
    
#Reset the counter variables for the animation function.
# for some reason if j isn't reset, it can get buggy       
j=0
k=0
#Create and save Visual #3 (Heatmap video)
ani = animation.FuncAnimation(fig3, animateheatmap, interval=1, save_count=len(Qit))

plt.rcParams['animation.ffmpeg_path'] ='C:\\Program Files\\Python\\ffmpeg-20200713-7772666-win64-static\\bin\\ffmpeg.exe'
FFwriter=animation.FFMpegWriter(fps=24, extra_args=['-vcodec', 'libx264'])
ani.save(fn_heatmap, writer=FFwriter)

plt.show()

print('Done')

#Automatically open visualization files
# startfile(viddir_path)
# startfile(figdir_eps)
# startfile(viddir_hm)

