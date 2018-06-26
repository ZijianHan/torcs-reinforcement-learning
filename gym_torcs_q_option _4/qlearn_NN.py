import random
import tensorflow as tf
import numpy as np

class QLearn_NN:
    def __init__(self, sess, num_state,num_action, lr=0.01,epsilon=0.1, gamma=0.9):
        self.sess = sess

        self.epsilon = epsilon  # exploration constant
        self.gamma = gamma
        self.actions = num_action

        self.s = tf.placeholder(tf.float32, [1, num_state], "state")
        self.v_ = tf.placeholder(tf.float32, [1, num_action], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Qnetwork'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=300,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=num_action,  # output units
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error
    def chooseAction(self,state):
        state = state[np.newaxis, :]
        if random.random() < self.epsilon:
            action = np.random.choice(self.actions,1)
            print("random choise is: ",action)
        else:
            v = self.sess.run(self.v, {self.s: state})
            action = tf.argmax(v,1)
        return action
