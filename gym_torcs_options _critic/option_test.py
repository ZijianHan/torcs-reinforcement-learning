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
from CriticNetwork import CriticNetwork
import snakeoil3_gym as snakeoil3

from gym_torcs_overtake import TorcsEnv

class IntraOptionPolicy:
    def __init__(self, sess, option):
        self.sess = sess
        self.option_behavior = option # 0 for overtaking, 1 for following
        self.actor = ActorNetwork(self.sess, 65, 3, 32, 0.001, 0.001)
        if option == 0:
            try:
                self.actor.model.load_weights("actormodel_overtaking.h5")
                self.actor.target_model.load_weights("actormodel_overtaking.h5")
                print("Overtaking Weight load successfully")
            except:
                print("Cannot find the overtaking weight")
        else:
            self.actor.model.load_weights("actormodel_following.h5")
            self.actor.target_model.load_weights("actormodel_following.h5")

        self.sess.run(tf.global_variables_initializer())

    def get_action(self, ob):
        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))
        state = s_t.reshape(1, s_t.shape[0])
        print("state feedin is :",state)
        a_t = self.actor.model.predict(state)
        print("Delta: {}, Target speed: {}".format(a_t[0][0],a_t[0][1]))
        a_t_primitive = Low_level_controller(a_t[0][0],a_t[0][1],ob, False)

        return a_t_primitive


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


if __name__ == '__main__':
    sess = tf.Session()

    args = parser.parse_args()

    env = TorcsEnv(vision=args.vision, throttle=True,gear_change=False)

    overtaking_policy = ActorNetwork(sess, args.state_size, args.action_size, args.batch_size, args.tau, args.learning_rate_actor)
    overtaking_policy.model.load_weights("actormodel_overtaking.h5")
    overtaking_policy.target_model.load_weights("actormodel_overtaking.h5")
    following_policy = ActorNetwork(sess, args.state_size, args.action_size, args.batch_size, args.tau, args.learning_rate_actor)
    following_policy.model.load_weights("actormodel_following.h5")
    following_policy.target_model.load_weights("actormodel_following.h5")
    option_policies = [overtaking_policy,following_policy]
    for episode in range(args.nepisodes):
        ob = env.reset(relaunch=True)
        state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))
        # Choose an option based on initial states
        option = 0
        # generate a first primitive action according to current option
        #action = option_policies[option].get_action(ob)
        action = option_policies[option].model.predict(state.reshape(1, state.shape[0]))
        action = Low_level_controller(action[0][0],action[0][1],ob, False)
        for step in range(args.nsteps):
            ob, r_t_primitive, done, _ = env.step(action)
            state_ = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))
            #action = option_policies[0].get_action(ob)
            action = option_policies[option].model.predict(state.reshape(1, state.shape[0]))
            action = Low_level_controller(action[0][0],action[0][1],ob, False)
            state = state_
            if done:
                break
    env,end()
    print("Finish.")
