import argparse
import numpy as np
import os
from PyQt5.QtWidgets import QApplication

import Environment
import sys
from algorithms.DQN import DQN
from algorithms.A2C import A2C
from utils import str2bool

# HYPERPARAMETERS
batch_size = 40
eps_start = 1
eps_end = 0.01
eps_decay = 0.0001
target_update = 10
memory_size = 10000

gamma = 0.999
lr = 0.0001
num_episodes = 2000
steps = 1500

arenaWidth = 3000  # cm entspricht 30m
arenaLength = 2000  # cm entspricht 20m

scaleFactor = 1.8

if __name__ == '__main__':
    args = None
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--manually', type=str2bool, nargs='?', const=True, default=False, help='Moving robot manually with wasd')
    parser.add_argument('--nb_episodes', type=int, default=num_episodes, help='Number of training episodes')
    parser.add_argument('--save_intervall', type=int, default=200, help='Save Intervall')
    parser.add_argument('--path', type=str, default='', help='Path where Models are saved')
    parser.add_argument('--alg', type=str, default='a2c', choices=['a2c', 'dqn'], help='Reinforcement Learning Algorithm')
    parser.add_argument('-lr', '--learningrate', type=float, default=lr, help='Learning Rate')
    parser.add_argument('--gamma', type=float, default=gamma, help='Gamma')
    parser.add_argument('--steps', type=int, default=steps, help='Steps in Environment per Episode')
    # FOR DQN
    parser.add_argument('--target_update', type=int, default=target_update, help='How often is the Agent updated')
    parser.add_argument('--batchsize', type=int, default=batch_size, help='batch_size')
    parser.add_argument('--memory_size', type=int, default=memory_size, help='Replay Memory Size')
    parser.add_argument('--eps_start', type=float, default=eps_start, help='Epsilon Start')
    parser.add_argument('--eps_end', type=float, default=eps_end, help='Epsilon End')
    parser.add_argument('--eps_decay', type=float, default=eps_decay, help='Epsilon Decay')

    parser.add_argument('--arena_width', type=int, default=arenaWidth, help='Width of the AI Arena')
    parser.add_argument('--arena_length', type=int, default=arenaLength, help='Length of the AI Arena')

    parser.add_argument('--scale_factor', type=int, default=scaleFactor, help='Scale Factor for visualisation')

    args = parser.parse_args(args)

    env_dim = (4, 9) #Timeframes, Robotstates

    app = QApplication(sys.argv)
    env = Environment.Environment(app, args.steps, args, env_dim[0])


    act_dim = np.asarray(env.get_actions()).shape[0]

    if args.path == "":
        args.path = os.path.join(os.getcwd(), "models", "")

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    if args.manually:
        args.steps = 1000000

    if args.alg == 'a2c':
        model = A2C(act_dim, env_dim, args)

    elif args.alg == 'dqn':
        model = DQN(act_dim, env_dim, args)

    model.train(env, args)

