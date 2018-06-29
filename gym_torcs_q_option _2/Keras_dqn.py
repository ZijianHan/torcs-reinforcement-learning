import os
import gym
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

from gym_torcs import TorcsEnv

# Superparameters
vision = False
OUTPUT_GRAPH = True
MAX_EPISODE = 2000
MAX_EP_STEPS = 20   # maximum time step in one episode
GAMMA = 0.999     # reward discount in TD error
EPSILON = 1.0#0.073
EPSILON_MIN = 0.05 #0.05
EPSILON_DECAY = 0.997
LR =0.001     # learning rate for critic
PI= 3.14159265359
step_time = [10,20,30,10,20,30]


def Low_level_controller(ob, safety_constrain,option):
    ob_angle = ob.angle
    ob_speedX = ob.speedX * 300

    # set targets for different options
    speed_target = 1
    if option == 0 or option==1 or option == 2:
        delta = 0.5
        for i in range(2):
            if ob.opponents[i+17] < 15/200:
                speed_target -= ((1.2/(200*ob.speedX))/ob.opponents[i+17])
                print("braking")
                break
    else:
        delta = -0.5
        for i in range(2):
            if ob.opponents[i+17] < 15/200:
                speed_target -= ((1.2/(200*ob.speedX))/ob.opponents[i+17])
                print("braking")
                break


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
    pAngleOffset = 5

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
    safety_distance_long = 8/200
    safety_distance_lat = 10/200
    #print(ob.opponents)
    #if option == 1:
    #    safety_constrain = 0

    if (safety_constrain):
        frontDis_list = []
        sideDis_list = []
        base_point = 17



        for i in range(2):
            if ob.opponents[i+base_point] < safety_distance_long:
                action_accel = 0
                action_brake = 0.4 #1.0*(25-min(frontDis_list) * 200)/15
                print("Frontal collision warning")
                break


        for j in range(11):
            if ob.opponents[j+22] < safety_distance_lat:
                #action_steer += 0.2
                action_steer += 0.3*(15-(ob.opponents[j+22] * 200))/15 # used to be 0.5 *
                print("Side collision warning")
                break
                '''
                if ob.opponents[j+8] < safety_distance_lat:
                    #action_steer += 0.2
                    action_steer -= 0.3*(15-(ob.opponents[j+22] * 200))/15 # used to be 0.5
            '''


    a_t = [action_steer, action_accel, action_brake]
    return a_t

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = GAMMA    # discount rate
        self.epsilon = EPSILON  # exploration rate
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LR
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()


    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(200, input_dim=self.state_size, activation='relu'))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))#loss = 'mse',
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, train_indicator):
        if train_indicator:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        print("action values are: ",act_values)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            #target_f = self.model.predict(state)
            #target_f[0][action] = target
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def playGame(train_indicator=0):
    plt.ion()
    plt.show()
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False)
    state_size = 29+36
    action_size = 6
    agent = DQNAgent(state_size, action_size)
    try:
        agent.load("save/six_options/torcs-dqn.h5")
    except:
        print("Cannot find the weight")
    done = False
    batch_size = 32
    if train_indicator:
        safety_constrain = 0
    else:
        safety_constrain = 1

    for i_episode in range(MAX_EPISODE):
        ob = env.reset(relaunch=True)
        '''
        if np.mod(i_episode, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()
        '''


        state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))
        state = np.reshape(state, [1, state_size])
        reward_ep = 0
        for step in range(MAX_EP_STEPS):
            option = agent.act(state,train_indicator)
            print("option is: ",option)
            reward = 0
            for i in range(step_time[option]):
                action = Low_level_controller(ob, False,option)
                print(action)
                ob, r_t_primitive, done, info = env.step(action)
                reward = reward + GAMMA**(i)*r_t_primitive
                if done:
                    break
            next_state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, option, reward, next_state, done)
            state = next_state
            reward_ep  += reward
            if done:
                break
        print("episode: {}/{}, score: {}, e: {:.2}"
              .format(i_episode, MAX_EPISODE, reward_ep, agent.epsilon))
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if i_episode % 10 == 0:
            if train_indicator:
                agent.save("save/six_options/torcs-dqn.h5")
        plt.figure(1)
        plt.hold(True)
        plt.subplot(211)
        plt.plot(i_episode,reward_ep,'ro')
        plt.xlabel('episode')
        plt.ylabel('Total reward per epsiode')
        plt.subplot(212)
        plt.hold(True)
        plt.plot(i_episode,agent.epsilon,'bo')
        plt.xlabel('episode')
        plt.ylabel('Epsilon')
        plt.draw()
        plt.show()
        plt.pause(0.001)
    env.end()  # This is for shutting down TORCS
    plt.savefig('test.png')
    print("Finish.")

if __name__ == "__main__":
    playGame()
