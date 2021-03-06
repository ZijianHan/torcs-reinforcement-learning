import os
import gym
import argparse
from config import *
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import h5py
from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
import snakeoil3_gym as snakeoil3

from gym_torcs import TorcsEnv

import csv
import scipy.io as sio

class OptionValueCritic:
    def __init__(self, state_size, option_size, discount, learning_rate_critic,epsilon,epsilon_min,epsilon_decay,tau):
        #self.sess = sess
        self.state_size = state_size
        self.option_size = option_size
        self.memory = deque(maxlen=2000)
        self.learning_rate = learning_rate_critic
        self.discount = discount #gamma
        self.options = option_size
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.TAU = tau

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        #self.sess.run(tf.global_variables_initializer())


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(300, input_dim=self.state_size, activation='relu'))
        model.add(Dense(600, activation='relu'))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(self.option_size))
        model.compile(loss = 'mse',
                      optimizer=Adam(lr=self.learning_rate))#loss = 'mse',
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.TAU * weights[i] + (1 - self.TAU)* target_weights[i]
        self.target_model.set_weights(target_weights)

    def remember(self, state, option, reward, next_state, done):
        self.memory.append((state, option, reward, next_state, done))

    def replay(self, batch):
        for state, option, reward, next_state, done in batch:
            state = np.reshape(state, [1, self.state_size])
            next_state = np.reshape(next_state, [1, self.state_size])
            update_target = self.model.predict(state)
            if done:
                update_target[0][option] = reward
            else:
                next_values = self.target_model.predict(next_state)
                update_target[0][option] = reward + self.discount*(np.max(next_values[0]))

            self.model.fit(state, update_target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        print("epsilon is: ",self.epsilon)

    def value(self, state, option=None):
        value = self.model.predict(state)[0]
        if option is None:
            return np.sum(value, axis=0)             # Q(s,:) = V(s)
        return np.sum(value[option], axis=0)            # Q(s,w)

    def advantage(self, state, option=None):
        value = self.model.predict(state)
        advantages = values - np.max(values)

        if option is None:
            return advantages
        return advantages[option]

    def get_option(self,state,train_indicator):
        act_values = self.target_model.predict(state) # get Q(s,:)
        print("action values are: ",act_values)
        if train_indicator:
            if np.random.rand() <= self.epsilon:
                print("random option!!!")
                return random.randrange(self.options)
            else:
                return np.argmax(act_values[0])  # returns action
        else:
            return np.argmax(act_values[0])  # returns action

    def terminate(self,state,option):
        value = self.model.predict(state)
        if value[0][option] == np.max(value[0]):
            return 0
        else:
            return 1

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
'''
class following_policy:
    def __init__(self,delta):
        self.delta = delta

    def act(self,ob):
        speed_target = 1

        for i in range(2):
            if ob.opponents[i+17] < 15/200:
                if ob.speedX <= 0.001:
                    speed_target = 1.0
                else:
                    speed_target -= ((1.2/(200*ob.speedX))/ob.opponents[i+17])
                print("braking")
                break

        if speed_target < 0:
            speed_target = 0.0
        elif speed_target > 1:
            speed_target = 1.0
        action = [[self.delta,speed_target]]
        return action
'''
def Low_level_controller(delta, speed_target, ob, safety_constrain):
    ob_angle = ob.angle
    ob_speedX = ob.speedX * 300
    lateralSetPoint = delta
    #if option == 1:
    #    lateralSetPoint += 0.3
    # Steer control
    if lateralSetPoint < -1:
        lateralSetPoint = -1
    elif lateralSetPoint > 1:
        lateralSetPoint = 1
    if speed_target < 0:
        speed_target = 0
    elif speed_target > 1:
        speed_target = 1

    pLateralOffset = 0.5
    pAngleOffset = 3

    action_steer = -pLateralOffset *(ob.trackPos + lateralSetPoint) + pAngleOffset * ob_angle


    action_steer = np.tanh(action_steer)

    # Throttle Control
    MAX_SPEED = 120
    MIN_SPEED = 10
    target_speed = MIN_SPEED + speed_target * (MAX_SPEED - MIN_SPEED)

    if ob_speedX > target_speed:
        action_brake = - 0.1 * (target_speed - ob_speedX)
        action_brake = np.tanh(action_brake)
        action_accel = 0
    else:
        action_brake = 0
        action_accel = 0.1 * (target_speed - ob_speedX)
        if ob_speedX < target_speed - (action_steer*50):
            action_accel+= .01
        if ob_speedX < 10:
           action_accel+= 1/(ob_speedX +.1)
        action_accel = np.tanh(action_accel)

    # Traction Control System
    if ((ob.wheelSpinVel[2]+ob.wheelSpinVel[3]) -
       (ob.wheelSpinVel[0]+ob.wheelSpinVel[1]) > 5):
       action_accel-= .2
    #safety_distance_long = (15 + 10* ob_speedX/MAX_SPEED)/200
    safety_distance_long = 10/200
    safety_distance_lat = 5/200
    #print(ob.opponents)
    #if option == 1:
    #    safety_constrain = 0

    if (safety_constrain):
        frontDis_list = []
        sideDis_list = []
        #if (ob.distFromStart < 1000-15 or (ob.distFromStart >1313 and ob.distFromStart<2313-15)):
        base_point = 17
        #else:
        #    base_point = 18


        for i in range(2):
            if ob.opponents[i+base_point] < safety_distance_long:
                action_accel = 0
                action_brake = 0.4 #1.0*(25-min(frontDis_list) * 200)/15
                print("Frontal collision warning")
                break


        for j in range(8):
            if ob.opponents[j+22] < safety_distance_lat:
                #action_steer += 0.2
                action_steer += ob.speedX*2.5*(5-(ob.opponents[j+22] * 200))/5 # used to be 0.5 *
                print("Side collision warning")
                break
                '''
                if ob.opponents[j+8] < safety_distance_lat:
                    #action_steer += 0.2
                    action_steer -= 0.3*(15-(ob.opponents[j+22] * 200))/15 # used to be 0.5
            '''
        '''
        for k in range(8):
            if ob.opponents[k+3] < safety_distance_lat:
                #action_steer += 0.2
                action_steer += ob.speedX*2*(5-(ob.opponents[k+22] * 200))/5 # used to be 0.5 *
                print("right Side collision warning")
                break
        '''

    a_t = [action_steer, action_accel, action_brake]

    return a_t

def playGame(train_indicator=1, safety_constrain_flag = True):    #1 means Train, 0 means simply Run
    plt.ion()
    args = parser.parse_args()

    np.random.seed(1337)

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    # Define two intra-policies
    overtaking_policy = ActorNetwork(sess, args.state_size, args.action_size)
    following_policy = ActorNetwork(sess, args.state_size, args.action_size)
    try:
        overtaking_policy.model.load_weights("actormodel_overtaking.h5")
        overtaking_policy.target_model.load_weights("actormodel_overtaking.h5")
        following_policy.model.load_weights("actormodel_following.h5")
        following_policy.target_model.load_weights("actormodel_following.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    # with fixed following policy
    #option_policies = [overtaking_policy,overtaking_policy,overtaking_policy,following_policy(0.5),following_policy(0.5),following_policy(0.5)]
    # with learned following policy
    option_policies = [overtaking_policy,overtaking_policy,overtaking_policy,following_policy,following_policy,following_policy]

    termination_steps = [10,20,30,10,20,30]

    # Define option-value function Q_Omega(s,omega): estimate values upon arrival
    critic = OptionValueCritic(args.state_size, args.option_size, args.discount, args.learning_rate_critic,args.epsilon,args.epsilon_min,args.epsilon_decay,args.tau)

    try:
        critic.load("option_value_model.h5")
        print("Critic Weight load successfully")
    except:
        print("Cannot find the critic weight")

    history = np.zeros((args.nepisodes, 2))

    # Define a buffer space to store samples
    buff = ReplayBuffer(args.buffer_size)    #Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=args.vision, throttle=True,gear_change=False)

    print("TORCS Experiment Start.")

    cumreward_list = []
    average_step_reward_list = []
    damage_rate_list = []
    epsilon_list = []
    results_list = []
    option_list = []
    trackPos_list = []
    speed_list = []
    epreward_list = []

    for episode in range(args.nepisodes):
        # Define variables to store values
        cumreward = 0.
        duration = 1
        option_switches = 0
        avgduration = 0.
        reward_option = 0
        total_options = 0
        damage_times = 0
        danger_time = 0
        collision_time = 0
        primitive_action_step = 0



        if np.mod(episode, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()


        state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))
        state = state.reshape(1, state.shape[0])

        for step in range(args.nsteps):
            total_options += 1
            option = critic.get_option(state,train_indicator)
            reward_option = 0
            for i in range(termination_steps[option]):
                primitive_action_step += 1
                action = option_policies[option].target_model.predict(state)
                '''
                if option == 0 or option == 1 or option == 2:
                    action = option_policies[option].target_model.predict(state)
                else:
                    action = option_policies[option].act(ob)
                '''
                print(action)
                action = Low_level_controller(action[0][0],action[0][1],ob, safety_constrain_flag)

                print("Option: {} Action:{}".format(option,action))
                ob, r_t_primitive, done, _ = env.step(action)
                if r_t_primitive == -30.0:
                    collision_time += 1
                elif r_t_primitive == -5.0:
                    danger_time += 1
                damage_times = collision_time + danger_time


                option_list.append(option)
                trackPos_list.append(ob.trackPos)
                speed_list.append(ob.speedX)
                epreward_list.append(r_t_primitive)

                reward_option = reward_option + args.discount**(i)*r_t_primitive
                state_ = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))
                state_ = state_.reshape(1, state_.shape[0])
                state = state_
                if done:
                    break


            buff.add(state,option,reward_option,state_,done)


            cumreward += reward_option

            reward_ep_per_step = cumreward/primitive_action_step
            damage_rate = damage_times/primitive_action_step
            if done:
                break
        if train_indicator:
            batch = buff.getBatch(args.batch_size)
            critic.replay(batch)
            if episode % 10 == 0:
                critic.save("option_value_model.h5")
        if train_indicator:
            # Save the results
            cumreward_list.append(cumreward)
            average_step_reward_list.append(reward_ep_per_step)
            damage_rate_list.append(damage_rate)
            epsilon_list.append(critic.epsilon)
            results_list = [cumreward_list,average_step_reward_list,damage_rate_list,epsilon_list]
            sio.savemat('results_both_learned.mat',{'total_reward':cumreward_list,'average_reward':average_step_reward_list,'epsilon':epsilon_list,'damage_rate':damage_rate_list})
        else:
            sio.savemat('test1lf1r.mat',{'ep_reward':epreward_list,'option':option_list,'trackPos':trackPos_list,'speed':speed_list})
            print('damage rate is:',damage_rate)

        history[episode, 0] = step
        history[episode, 1] = avgduration






        plt.figure(1)
        plt.hold(True)
        plt.subplot(311)
        plt.plot(episode,cumreward,'ro')
        plt.xlabel('episode')
        plt.ylabel('Total reward per epsiode')
        plt.subplot(312)
        plt.hold(True)
        plt.plot(episode,cumreward/total_options,'bo')
        plt.xlabel('episode')
        plt.ylabel('Average reward per option')
        plt.subplot(313)
        plt.hold(True)
        plt.plot(episode,critic.epsilon,'go')
        plt.xlabel('episode')
        plt.ylabel('epsilon')

        plt.draw()
        plt.show()
        plt.pause(0.001)



    env.end()  # This is for shutting down TORCS
    plt.savefig('test.png')

    print("Finish.")


if __name__ == '__main__':
    playGame()
