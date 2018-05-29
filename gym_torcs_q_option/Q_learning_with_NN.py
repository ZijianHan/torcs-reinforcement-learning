import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from gym_torcs_overtake import TorcsEnv
from qlearn_NN import QLearn_NN

# Superparameters
vision = False
OUTPUT_GRAPH = True
MAX_EPISODE = 3000
MAX_EP_STEPS = 15   # maximum time step in one episode
GAMMA = 0.9     # reward discount in TD error
EPSILON = 0.5
LR = 0.01     # learning rate for critic
PI= 3.14159265359
step_time = 30

def Get_actions(o_t, ob):
    if o_t == 0:
        offset = -0.5
    else:
        offset = 0.5
    lateralSetPoint = offset
    pLateralOffset = 0.6
    pAngleOffset = 12

    action_steer = -pLateralOffset *(ob.trackPos + lateralSetPoint) + pAngleOffset * ob.angle

    #action_steer = np.tanh(action_steer)
    a_t_primitive = [action_steer]

    return a_t_primitive

def playGame(train_indicator=1):
    # Load the environment
    plt.ion()
    plt.show()
    env = TorcsEnv(vision=vision, throttle=False,gear_change=False)

    tf.reset_default_graph()

    num_state = 29+36
    num_action = 2

    sess = tf.Session()

    Qnet = QLearn_NN(sess, num_state,num_action, LR,EPSILON, GAMMA)

    sess.run(tf.global_variables_initializer())


    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    # create lists to contain total rewards and steps per episode
    jList = [] # step list
    rList = [] # reward each episode
    errorList = []


    for i_episode in range(MAX_EPISODE):
        print(i_episode)

        rAll = 0  # total reward for each episode
        td_error = 0 # total td error
        done = False # done
        j = 0 # steps each episode
        if np.mod(i_episode, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))

        while j<MAX_EP_STEPS:
            j+=1

            a = Qnet.chooseAction(s_t)
            print(a)
            r = 0
            for i in range(step_time):
                action_steer = Get_actions(a,ob)

                ob, r_t_primitive, done, info = env.step(action_steer)
                r = r_t_primitive + GAMMA*r
                if done:
                    break

            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))

            if train_indicator:
                td_error += Qnet.learn(s_t,r,s_t1)

            rAll += GAMMA*r
            s_t = s_t1
            if done == True:
                #Reduce chance of random action as we train the model.
                e = 1./((i_episode/50) + 10)
                break

        jList.append(j)
        rList.append(rAll)
        errorList.append(td_error)

        print("Percent of succesful episodes: " + str(sum(rList)/MAX_EPISODE) + "%")

        # Plot some statistics on network performance
        plt.figure(1)
        plt.plot(rList)
        plt.figure(2)
        plt.plot(jList)
        #plt.figure(3)
        #plt.plot(errorList)
        plt.draw()
        plt.show()

    env.end()  # This is for shutting down TORCS

if __name__ == "__main__":
    playGame()
