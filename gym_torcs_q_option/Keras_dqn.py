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

from gym_torcs_overtake import TorcsEnv

# Superparameters
vision = False
OUTPUT_GRAPH = True
MAX_EPISODE = 1000
MAX_EP_STEPS = 15   # maximum time step in one episode
GAMMA = 0.999     # reward discount in TD error
EPSILON = 0.0#0.073
EPSILON_MIN = 0.0 #0.05
EPSILON_DECAY = 0.995
LR = 0.001     # learning rate for critic
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

    def act(self, state):
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

def playGame(train_indicator=1):
    plt.ion()
    plt.show()
    env = TorcsEnv(vision=vision, throttle=False,gear_change=False)
    state_size = 29+36
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    agent.load("save/torcs-dqn.h5")
    done = False
    batch_size = 32

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
            action = agent.act(state)
            reward = 0
            for i in range(step_time):
                action_steer = Get_actions(action,ob)

                ob, r_t_primitive, done, info = env.step(action_steer)
                reward = reward + GAMMA**(i)*r_t_primitive
                if done:
                    break
            next_state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            reward_ep  += reward
            if done:
                break
        print("episode: {}/{}, score: {}, e: {:.2}"
              .format(i_episode, MAX_EPISODE, reward_ep, agent.epsilon))
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        #if i_episode % 10 == 0:
        #    agent.save("save/torcs-dqn.h5")
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
