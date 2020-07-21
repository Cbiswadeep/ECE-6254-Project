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
########################## ADDED BY JASON #####################################
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
fn = 'QL_Iterations.mp4'    #filename for iteration video
######################## END ADDED BY JASON ###################################

# Environment
####################### EDITED BY JASON ######################################
custom_map = ['SFFF','FHFH','FFFH','HFFG']  # added by Jason - standard 4x4 map
env = gym.make('FrozenLake-v0', is_slippery=True, desc=custom_map)
print("")
print('Lake Visualization:')
print('S=Start, F=Frozen, H=Hole, G=Goal')
env.render()
##################### END EDITED BY JASON ####################################
inputCount = env.observation_space.n
actionsCount = env.action_space.n

########################## ADDED BY JASON #####################################
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

# Functions for visualization
vizit = 250     #number of iterations between visualizations
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
######################## END ADDED BY JASON ###################################

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

########################## ADDED BY JASON #####################################
#update list of moves to say it reset
    if i == 0 or (i+1) % vizit == 0 or i == episodes-1:
        if i == 0:
            moves = [-1]
            itnum = [1]
        else:
            moves.append(-1)
            itnum.append(i+1)
######################## END ADDED BY JASON ###################################


    while not done:
        if np.random.random() < epsilon:
            a = np.random.randint(0, actionsCount)
        else:
            a = np.argmax(Q[s])

        newS, r, done, _ = env.step(a)
        Q[s][a] = Q[s][a] + lr * (r + gamma * np.max(Q[newS]) - Q[s][a])
        
########################## ADDED BY JASON #####################################
        #Update moves array
        if i == 0 or (i+1) % vizit == 0 or i == episodes-1:
            moves = movefun(s,newS,moves)
            itnum.append(i+1)
######################## END ADDED BY JASON ###################################

        s = newS

        if lr > lrMin:
            lr *= lrDecay

        if not r==0 and epsilon > epsilonMin:
            epsilon *= epsilonDecay


print("")
print("Learning Rate :", lr)
print("Epsilon :", epsilon)

########################## ADDED BY JASON #####################################
#Create lake graphic
fig,ax = plt.subplots()
plt.xlim(0,ncol)
plt.ylim(0,nrow)
plt.axis('square')
colormap = ListedColormap(['g','b','c','y'])
c = ax.pcolor(np.linspace(0,ncol,ncol+1),np.linspace(0,nrow,nrow+1),list(lake[::-1]),cmap=colormap,alpha=0.4)       

x, y = robotloc(moves,0.5,nrow-.5)
line, = ax.plot(x[0],y[0],'ko-')
point, = ax.plot(x[0],y[0],'r*',markersize=15)
plt.title('Iteration 0')

j=0
k=0
def animate(j):
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

ani = animation.FuncAnimation(fig, animate, interval=1, save_count=len(x))

plt.rcParams['animation.ffmpeg_path'] ='C:\\Program Files\\Python\\ffmpeg-20200713-7772666-win64-static\\bin\\ffmpeg.exe'
FFwriter=animation.FFMpegWriter(fps=10, extra_args=['-vcodec', 'libx264'])
ani.save(fn, writer=FFwriter)

plt.show()
######################## END ADDED BY JASON ###################################


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

########################## ADDED BY JASON #####################################
#Automatically open video file
import os
from os import startfile
cdir = os.path.abspath(os.getcwd())
viddir = os.path.join(cdir,fn)
startfile(viddir)
######################## END ADDED BY JASON ###################################
