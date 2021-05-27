import argparse
import numpy as np
import os
from PyQt5.QtWidgets import QApplication

import yaml

import EnvironmentWithUI
import sys
import h5py
import datetime
from algorithms.DQN import DQN
from algorithms.A2C_parallel.A2C_Multi import A2C_Multi
from algorithms.A2C_parallel.PPO_Multi import PPO_Multi
from algorithms.utils import str2bool

# HYPERPARAMETERS
batch_size = 40
eps_start = 1
eps_end = 0.01
eps_decay = 0.0001
target_update = 10
memory_size = 10000

gamma = 0.999               # discount factor for calculating the discounted reward
lr = 0.0003                 # learning rate
num_episodes = 2500         # the number of epochs (/episodes) that are simulated
steps = 600                 # number of steps per epoch (/episode)
trainingInterval = 75       # number of steps after which the neural net is trained

arenaWidth = 22             # Width (X Direction) of the Arena in Meter
arenaLength = 10            # Length (Y direction) of the Arena in meter
simTimeStep = 0.15          # simulated time between two steps in the simulation

angleStepsSonar = 0.5       # spacing between two light rays (for distance calculation) in degrees
timeFrames = 4              # number of past states used as an Input for the neural net
numbOfRobots = 4            # only change if set to manual do not use more than 4
numbOfParallelEnvs = 3*8    # parallel environments are used to create more and diverse training experiences

scaleFactor = 65            # scales the simulation window (the window is also rezisable, only change if your display is low res)

startTime = datetime.datetime.now().strftime("_%y-%m-%d--%H-%M")  # Timestamp used for saving the model

filename = "" # enter the filename from the models folder (without .h5 or .yml)
filename = 'PPO_21-05-07--12-26_e580'
filename = 'PPO_21-05-10--14-58'
# filename = 'PPO_21-05-14--10-10'


if __name__ == '__main__':
    args = None
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--manually', type=str2bool, nargs='?', const=True, default=False, help='Moving robot manually with wasd')
    parser.add_argument('--nb_episodes', type=int, default=num_episodes, help='Number of training episodes')
    parser.add_argument('--save_intervall', type=int, default=50, help='Save Intervall')
    parser.add_argument('--path', type=str, default='', help='Path where Models are saved')
    parser.add_argument('--model_timestamp', type=str, default=startTime, help='Timestamp from when the model was created')
    parser.add_argument('--alg', type=str, default='ppo', choices=['a2c', 'dqn', 'ppo'], help='Reinforcement Learning Algorithm')
    parser.add_argument('-lr', '--learningrate', type=float, default=lr, help='Learning Rate')
    parser.add_argument('--gamma', type=float, default=gamma, help='Gamma')
    parser.add_argument('--steps', type=int, default=steps, help='Steps in Environment per Episode')
    parser.add_argument('--train_interval', type=int, default=trainingInterval, help='The number of steps after which the neural net is trained.')
    parser.add_argument('--net_size', type=str, default='big', choices=['small', 'medium', 'big'], help='Determines the number of filters in the convolutional layers, the overall amount of neurons and the number of layers.')
    parser.add_argument('--shared', type=str2bool, default='False', help='Determines whether actor and aritic share their main network weights.')

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

    parser.add_argument('--mode', type=str, default='sonar', choices=['global', 'sonar'], help='Training Mode')  # Global oder Sonar einstellbar
    parser.add_argument('--collide_other_targets', type=str2bool, default=False, help='Determines whether the robot collides with targets of other robots (or passes through them)')  # Global oder Sonar einstellbar
    parser.add_argument('--time_penalty', type=str2bool, default='False', help='Reward function with time step penalty')
    parser.add_argument('--angle_steps', type=int, default=angleStepsSonar, help='Angle Steps for sonar training')
    parser.add_argument('--time_frames', type=int, default=timeFrames, help='Number of Timeframes which will be analyzed by neural net')
    parser.add_argument('--parallel_envs', type=int, default=numbOfParallelEnvs, help='Number of parallel environments used during training in addition to the main training process')
    parser.add_argument('--numb_of_robots', type=int, default=numbOfRobots, help='Number of robots acting in one environment')
    parser.add_argument('--sim_time_step', type=float, default=simTimeStep, help='Time between steps')

    parser.add_argument('--training', type=bool, default=True, help='Training or Loading trained weights')
    parser.add_argument('--use_gpu', type=bool, default=False, help='Use GPUS with Tensorflow (Cuda 10.1 is needed)')
    parser.add_argument('--load_old', type=bool, default=True, help='Improve existing net (by loading pretrained weights and continuing with training)')

    args = parser.parse_args(args)

    if args.path == "":
        args.path = os.path.join(os.getcwd(), "models", "")

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    if args.manually:
        args.steps = 1000000
        args.parallel_envs = 1

    if args.load_old or not args.training:       #FÜR CHRISTIANS NETZ GEWICHTE AUSKOMMENTIWER
        # Read YAML file
        with open(args.path + filename + ".yml", 'r') as stream:
            data_loaded = yaml.load(stream, Loader=yaml.UnsafeLoader)
            loadedArgs = data_loaded[0]
            args.steps = loadedArgs.steps
            args.time_frames = loadedArgs.time_frames
            args.time_penalty = loadedArgs.time_penalty
            args.angle_steps = loadedArgs.angle_steps
            args.net_size = loadedArgs.net_size
            args.shared = loadedArgs.shared
            #args.sim_time_step=loadedArgs.sim_time_step



    if args.mode == 'sonar':
        states = int((360 / angleStepsSonar) + 7)
        #states = int((1081) + 7) #FÜR CHRISTIANS NETZ GEWICHTE
        env_dim = (args.time_frames, states)  # Timeframes, Robotstates

    elif args.mode == 'global':
        env_dim = (4, 9)


    # app = QApplication(sys.argv)
    # env = EnvironmentWithUI.Environment(app, args, env_dim[0], 0)


    act_dim = np.asarray(2)#env.get_actions()) #TODO bei kontinuierlichem 2 actions

    if args.alg == 'a2c':
        model = A2C_Multi(act_dim, env_dim, args)
        # model = PPO_Multi(act_dim, env_dim, args)
        # model = A2C(act_dim, env_dim, args)
    elif args.alg == 'dqn':
        model = DQN(act_dim, env_dim, args)
    elif args.alg == 'ppo':
        model = PPO_Multi(act_dim, env_dim, args)

    if args.training:
        if args.load_old:
            model.train(args.path+filename+'.h5')
        model.train()
        # model.trainA3C()
    elif not args.training:
        app = QApplication(sys.argv)
        env = EnvironmentWithUI.Environment(app, args, env_dim[0])
        model.load_net(args.path+filename+'.h5')
        model.execute(env, args)




