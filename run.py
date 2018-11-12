#!/usr/bin/env python

import argparse
from time import sleep

from unityagents import UnityEnvironment

from maddpg import MADDPGAgent
from config import config
from train import training_loop
from play import play_loop


parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', dest='train', help='Set the train mode')
parser.add_argument('--file_prefix', default=None, help='Set the file for agent to load weights with using prefix')
parser.add_argument('--playthroughs', default=10, type=int, help='Number of playthroughs played in a play mode')
parser.add_argument('--sleep', default=0, type=int, help='Time before environment starts in a play mode [seconds]')
arguments = parser.parse_args()

env = UnityEnvironment(file_name='./Tennis.app', seed=config.general.seed)
brain_name = env.brain_names[0]
agent = MADDPGAgent(config=config, file_prefix=arguments.file_prefix)

if arguments.train:
    print('Train mode \n')
    training_loop(env, brain_name, agent, config)
else:
    print('Play mode \n')
    sleep(arguments.sleep)
    play_loop(env, brain_name, agent, playthrougs=arguments.playthroughs)
