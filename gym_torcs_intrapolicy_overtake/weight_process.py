import numpy as np
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

BUFFER_SIZE = 100000
BATCH_SIZE = 32
GAMMA = 0.9999
TAU = 0.001     #Target Network HyperParameters
LRA = 0.0001    #Learning rate for Actor
LRC = 0.001     #Lerning rate for Critic

action_dim = 2  #Steering/Acceleration/Brake
state_dim = 29+36  #of sensors input

#Tensorflow GPU optimization
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)

actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)

actor.model.load_weights("actormodel.h5")

weight_actor = actor.model.trainable_weights

print(weight_actor)
