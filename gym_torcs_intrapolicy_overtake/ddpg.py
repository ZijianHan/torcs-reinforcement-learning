from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
#from keras.engine.training import collect_trainable_weights
import json
import h5py
import matplotlib.pyplot as plt
import os

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
import snakeoil3_gym as snakeoil3

OU = OU()       #Ornstein-Uhlenbeck Process
def Get_actions(delta, speed_target, ob, safety_constrain = True):
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
        if (ob.distFromStart < 1000-15 or (ob.distFromStart >1313 and ob.distFromStart<2313-15)):
            base_point = 17
        else:
            base_point = 18


        for i in range(2):
            if ob.opponents[i+base_point] < safety_distance_long:
                action_accel = 0
                action_brake = 0.4 #1.0*(25-min(frontDis_list) * 200)/15
                print("Frontal collision warning")
                break


        for j in range(8):
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
    #print('actual speed is:',ob_speedX)
    #print('target_speed is:',speed_target, ';',target_speed)
    #print('steer:', action_steer,delta)
    #print('accel:',action_accel)
    #print('brake:',action_brake)
    return a_t
def playGame(train_indicator=0, safety_constrain_flag = False):    #1 means Train, 0 means simply Run
    initialization = 0
    episode_trained = 0
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.9999
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 2  #Steering/Acceleration/Brake
    state_dim = 29+36  #of sensors input

    np.random.seed(1337)

    vision = False

    EXPLORE = 100000.
    episode_count = 1000
    max_steps = 200
    reward = 0
    done = False
    step = 0
    epsilon = 0.4 #1
    indicator = 0

    plt.ion()


    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False)

    #Now load the weight
    print("Now we load the weight")
    if (initialization):
        print("Now we save model")
        actor.model.save_weights("actormodel.h5", overwrite=True)
        with open("actormodel.json", "w") as outfile:
            json.dump(actor.model.to_json(), outfile)

        critic.model.save_weights("criticmodel.h5", overwrite=True)
        with open("criticmodel.json", "w") as outfile:
            json.dump(critic.model.to_json(), outfile)

    try:
        actor.model.load_weights("actormodel_overtaking.h5")
        critic.model.load_weights("criticmodel_overtaking.h5")
        actor.target_model.load_weights("actormodel_overtaking.h5")
        critic.target_model.load_weights("criticmodel_overtaking.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("TORCS Experiment Start.")

    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        print("Epsilon is: ", epsilon)
        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()


        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))

        total_reward = 0.
        damage_steps = 0
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])

            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            noise_t[0][0] = train_indicator * max(epsilon, 0.0) * OU.function(a_t_original[0][0],  0.0 , 0.80, 0.80)
            #noise_t[0][1] = train_indicator * max(epsilon, 0.0) * OU.function(a_t_original[0][1],  1.0 , 1.00, 0.10)
            noise_t[0][1] = train_indicator * max(epsilon, 0.0) * OU.function(a_t_original[0][1],  0.9 , 1.0, 0.10)

            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)
            '''
            if np.random.randn() < max(epsilon,0.05):
                a_t[0][0] = np.random.randn()*2-1
            else:
                a_t[0][0] = a_t_original[0][0]
            '''
            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]

            a_t_primitive = Get_actions(a_t[0][0],a_t[0][1],ob, safety_constrain = safety_constrain_flag)

            ob, r_t, done, info = env.step(a_t_primitive)

            if r_t == -1:
                damage_steps += 1

            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm,ob.opponents))

            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer

            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]

            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                break

        damage_rate = (float)(damage_steps/j*100)

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel_overtaking.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel_overtaking.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        plt.figure(1)
        plt.hold(True)
        plt.subplot(411)
        plt.plot(i,total_reward,'ro')
        plt.xlabel("Episodie")
        plt.ylabel("Episodic total reward")
        plt.subplot(412)
        plt.plot(i,total_reward/j,'bo')
        plt.xlabel("Episodie")
        plt.ylabel("Expected reward each step")
        plt.subplot(413)
        plt.plot(i,damage_rate,'go')
        plt.xlabel("Episodie")
        plt.ylabel("Damage rate per episode [%]")
        plt.subplot(414)
        plt.plot(i,max(epsilon,0),'yo')
        plt.xlabel("Episodie")
        plt.ylabel("epsilon")
        plt.draw()
        plt.show()
        plt.pause(0.001)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    plt.savefig('test.png')
    print("Finish.")

if __name__ == "__main__":
    playGame()
