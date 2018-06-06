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
from OvertakingNetwork import OvertakingNetwork
from FollowingNetwork import FollowingNetwork
from ActorNetwork import ActorNetwork
import snakeoil3_gym as snakeoil3

from gym_torcs_overtake import TorcsEnv


class OptionValueCritic:
    def __init__(self, state_size, option_size, discount, learning_rate_critic,epsilon,epsilon_min,epsilon_decay):
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

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        #self.sess.run(tf.global_variables_initializer())


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(200, input_dim=self.state_size, activation='relu'))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(self.option_size))
        model.compile(loss = 'mse',
                      optimizer=Adam(lr=self.learning_rate))#loss = 'mse',
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

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
                next_values = self.model.predict(next_state)
                if next_values[0][option] == np.max(next_values):
                    next_termination = 0
                else:
                    next_termination = 1
                update_target[0][option] = reward + self.discount*((1. - next_termination)*next_values[0][option] + next_termination*np.max(next_values[0]))

            self.model.fit(state, update_target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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

    def get_option(self,state):
        act_values = self.model.predict(state) # get Q(s,:)
        print("action values are: ",act_values)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.options)
        else:
            return np.argmax(act_values[0])  # returns action

    def terminate(self,state,option):
        value = self.model.predict(state)
        if value[0][option] == np.max(value):
            return 0
        else:
            return 1

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def Low_level_controller(delta, speed_target, ob, safety_constrain = True):
    ob_angle = ob.angle
    ob_speedX = ob.speedX * 300
    lateralSetPoint = delta
    # Steer control==
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
    safety_distance_long = 15/200
    safety_distance_lat = 15/200
    #print(ob.opponents)

    if (safety_constrain):
        for i in range(6):
            if ob.opponents[i+14] < safety_distance_long:
                action_accel = 0
                action_brake = 0.2
                print("Frontal collision warning")

        for j in range(8):
            if ob.opponents[j+22] < safety_distance_lat:
                #action_steer += 0.2
                action_steer += 0.5*(15-(ob.opponents[j+22] * 200))/15
                print("Side collision warning")

    a_t = [action_steer, action_accel, action_brake]

    return a_t

def playGame(train_indicator=0, safety_constrain_flag = False):    #1 means Train, 0 means simply Run
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
    overtaking_policy = OvertakingNetwork(sess, args.state_size, args.action_size)
    following_policy = FollowingNetwork(sess, args.state_size, args.action_size)
    try:
        overtaking_policy.model.load_weights("actormodel_overtaking.h5")
        overtaking_policy.target_model.load_weights("actormodel_overtaking.h5")
        following_policy.model.load_weights("actormodel_following.h5")
        following_policy.target_model.load_weights("actormodel_following.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    option_policies = [overtaking_policy, following_policy]

    history = np.zeros((args.nepisodes, 2))

    # Define a buffer space to store samples
    buff = ReplayBuffer(args.buffer_size)    #Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=args.vision, throttle=True,gear_change=False)

    print("TORCS Experiment Start.")

    for episode in range(args.nepisodes):
        # Define variables to store values
        cumreward = 0.
        duration = 1
        option_switches = 0
        avgduration = 0.
        reward_option = 0
        damage_steps = 0

        if np.mod(episode, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()


        state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))
        # Choose an option based on initial states
        option = 0 #critic.get_option(state.reshape(1, state.shape[0])) # policy.predict or sample (states)

        # generate a first primitive action according to current option

        for step in range(args.nsteps):
            action = option_policies[1].model.predict(state.reshape(1, state.shape[0]))
            action = Low_level_controller(action[0][0],action[0][1],ob, safety_constrain_flag)

            print("Step:{} Option: {} Action:{}".format(step,option,action))
            ob, r_t_primitive, done, _ = env.step(action)
            state_ = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))
            state = state_

            cumreward += r_t_primitive
            duration += 1
            if done:
                break



    env.end()  # This is for shutting down TORCS

    print("Finish.")


if __name__ == '__main__':
    playGame()
