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

from scipy.special import expit
from scipy.misc import logsumexp

from gym_torcs_overtake import TorcsEnv

class PolicyOverOption:
    def __init__(self,option_size,epsilon,epsilon_min,epsilon_decay,critic):
        self.options = option_size
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.critic = critic

    def get_option(self,state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.options)
        act_values = self.critic.value(state) # get Q(s,:)
        print("action values are: ",act_values)
        return np.argmax(act_values[0])  # returns action


class IntraOptionPolicy(object): #DDPG
    def __init__(self, sess, state_size, action_size,gamma=0.999,learning_rate_intrapolicy):
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma    # discount rate
        self.learning_rate = learning_rate_intrapolicy
        self.actor_state_input,self.actor_model = self._build_model()
        self.actor_critic_grad = tf.placeholder(tf.float32,[None, self.state_size]) # where we will feed de/dC (from critic)

		actor_model_weights = self.actor_model.trainable_weights
		self.actor_grads = tf.gradients(self.actor_model.output,actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
		grads = zip(self.actor_grads, actor_model_weights)
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        state_input = layers.Input(shape=(self.state_size,))
        h1 = Dense(200, activation='relu')(state_input)
		h2 = Dense(200, activation='relu')(h1)
		output = Dense(self.action_size, activation='tanh')(h2)
		model = Model(input=state_input, output=output)
		adam  = Adam(lr=self.learning_rate)
		model.compile(loss="mse", optimizer=adam)
		return state_input,model

    def train(self, sample,critic_grad):
        cur_state, action, reward, new_state, _ = sample
		self.sess.run(self.optimize, feed_dict={
			self.actor_state_input: cur_state,
			self.actor_critic_grad: grads
		})

    def get_action(self, state):
        self.epsilon *= self.epsilon_decay
		if np.random.random() < self.epsilon:
			return random.randrange(self.action_size) # here should generate random numbers
		return self.actor_model.predict(state)

class Termination(object):
    def __init__(self, sess, state_size,gamma=0.999,learning_rate_termination):
        self.sess = sess
        self.state_size = state_size
        self.action_size = 1
        self.gamma = gamma    # discount rate
        self.learning_rate = learning_rate_termination

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.float32, None, name="act")
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")  # TD_error

        with tf.variable_scope('TerminationNetwork'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=200,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            l2 = tf.layers.dense(
                inputs=l1,
                units=200,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )

            ter_prob = tf.layers.dense(
                inputs=l2,
                units=1,  # number of hidden units
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='ter_prob'
            )
        with tf.variable_scope('exp_v'):
            log_prob = tf.log(ter_prob)
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.exp_v)  # minimize(-exp_v) = maximize(exp_v)


    def train(self, state,critic_grad):
		self.sess.run(self.optimize, feed_dict={
			self.actor_state_input: state,
			self.actor_critic_grad: grads
		})

    def get_action(self, state):
        self.epsilon *= self.epsilon_decay
		if np.random.random() < self.epsilon:
			return random.randrange(2)
		return int(np.random.random() < self.actor_model.predict(state))

    def predict(self,state):
        return self.actor_model.predict(state)

class TerminationGradient:
    def __init__(self, terminations, critic):
        self.terminations = terminations
        self.critic = critic

    def update(self, state, option, td_error):
        self.terminations[option].train(state,td_error)

class OptionValueCritic:# learn Q_(s,w)
    def __init__(self, sess, state_size, option_size, discount, learning_rate_critic, terminations):
        self.sess = sess
        self.lr = learning_rate_critic
        self.discount = discount
        self.terminations = terminations
        self.options = option_size
        with tf.name_scope('inputs'):
            self.s = tf.placeholder(tf.float32, [1, num_state], "state")
            self.v_ = tf.placeholder(tf.float32, [1, num_action], "v_next")
            self.r = tf.placeholder(tf.float32, None, 'r')
            self.done = tf.placeholder(tf.bool,None,'done')

        with tf.variable_scope('Qnetwork'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=200,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            l2 = tf.layers.dense(
                inputs=l1,
                units=200,  # output units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )

            self.v = tf.layers.dense(
                inputs=l2,
                units=option_size,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='Q(s,w)'
            )

        with tf.variable_scope('squared_TD_error'):
            update_target = self.r
            if not self.done:
                current_values = self.v
                termination = self.terminations[self.last_option].predict(self.s)
                update_target += self.discount*((1. - termination)*current_values[self.last_option] + termination*np.max(current_values))
                # update_target += (1 - (1-self.discount)/self.priority) * (1. - termination)*current_values[self.last_option] + self.discount*termination*np.max(current_values)  # TODO

            # Dense gradient update step
            self.td_error = update_target - self.last_value

            if not self.done:
                self.last_value = current_values[option]
            self.last_option = option
            self.last_state = phi

            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)


    def start(self, state, option):
        self.last_state = state
        self.last_option = option
        v = self.sess.run(self.v, {self.s: state})
        self.last_value = v[self.last_option]

    def value(self, state, option=None):
        value = self.sess.run(self.v, {self.s: state})
        if option is None:
            return np.sum(value, axis=0)             # Q(s,:)
        return np.sum(value[option], axis=0)            # Q(s,w)

    def learn(self, s, r, s_,done):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r, self.done: done})
        return td_error

    def advantage(self, state, option=None):
        values = self.sess.run(self.v, {self.s: state})
        advantages = values - np.max(values)
        # advantages = (1 - (1-self.discount)/self.priority) * values - self.discount * np.max(values)        # TODO
        if option is None:
            return advantages
        return advantages[option]




class IntraOptionPolicy:
    def __init__(self, sess, option):
        self.sess = sess
        self.option_behavior = option # 1 for overtaking, 2 for following

    def act(self, state):
        if option == 0:
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




if __name__ == '__main__':

    sess = tf.Session()

    args = parser.parse_args()

    rng = np.random.RandomState(1234) # for random number generation

    env = TorcsEnv(vision=args.vision, throttle=True,gear_change=False)


    history = np.zeros((args.nepisodes, 2))

    overtaking_policy = IntraOptionPolicy(sess,1)

    following_policy = IntraOptionPolicy(sess,2)
    # Define intra-option policies: fixed policy
    option_polcies = [overtaking_policy, following_policy]

    # Define termination function for each option: linear sigmoid functions
    option_terminations = [Termination(sess, args.state_size,args.discount,args.learning_rate_termination) for _ in range(args.noptions)]

    # Define option-value function Q_Omega(s,omega): estimate values upon arrival
    critic = OptionValueCritic(sess, args.state_size, args.option_size, args.discount, args.learning_rate_critic, option_terminations)

    # Define policy over options: softmax option policy
    policy = PolicyOverOption(args.noptions,args.epsilon,args.epsilon_min,args.epsilon_decay,critic)
    #policy = SoftmaxPolicy(rng, nfeatures, args.noptions, args.temperature)

    # Define option-action-value function Q_U(s,w,a): intra-option critic (not really needed here)
    #action_critic = IntraOptionActionQLearning()

    # Improvement of the termination functions based on gradients
    termination_improvement= TerminationGradient(option_terminations, critic)


    for episode in range(args.nepisodes):
        ob = env.reset(relaunch=True)
        state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents, ob.racePos))

        # Choose an option based on initial states
        option = policy.sample(state) # policy.predict or sample (states)
        # generate a first primitive action according to current option
        action = option_policies[option].get_actions(state)
        # action = option_policies[option].sample(phi)  # put this to primitive action loop
        critic.start(phi, option)
        action_critic.start(phi, option, action)

        cumreward = 0.
        duration = 1
        option_switches = 0
        avgduration = 0.

        for step in range(args.nsteps):
            ob, reward, done, _ = env.step(action)
            state_ = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents, ob.racePos))

            # Termination might occur upon entering the new state
            if option_terminations[option].sample(phi):
                # if terminates, get a new option based on current state
                option = policy.get_action(state_)
                option_switches += 1
                avgduration += (1./option_switches)*(duration - avgduration)
                duration = 1

            # Generate primitive action for next step based on current intra-option policy
            action = option_policies[option].get_actions(state_)

            # Update Option-value function and Option-action-value fuction
            td_error = critic.learn(state, reward, state_, done)

            # Termination update
            termination_improvement.update(state, option, td_error)

            state = state_

            cumreward += reward
            duration += 1
            if done:
                break

        history[episode, 0] = step
        history[episode, 1] = avgduration

    print('nstep {}   duration {}'.format(np.mean(history[:,0]), np.mean(history[:,1])) )
