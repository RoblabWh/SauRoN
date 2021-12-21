import argparse
import math
import numpy as np
import os
import ray
from PyQt5.QtWidgets import QApplication
import yaml
import sys
import datetime

from ControlWindow import ControlWindow
from deprecated.A2C_parallel_old.A2C_Multi import A2C_Multi
from algorithms.PPO_parallel.PPO_Multi import PPO_Multi
from algorithms.utils import str2bool


# HYPERPARAMETERS
batch_size = 40
eps_start = 1
eps_end = 0.01
eps_decay = 0.0001
target_update = 10
memory_size = 10000

gamma = 0.999               # discount factor for calculating the discounted reward
lr = 0.0001                 # learning rate
num_episodes = 601         # the number of epochs (/episodes) that are simulated
steps = 750                 # number of steps per epoch (/episode)
trainingInterval = 75       # number of steps after which the neural net is trained

arenaWidth = 22             # Width (X Direction) of the Arena in Meter
arenaLength = 10            # Length (Y direction) of the Arena in Meter
simTimeStep = 0.15 #0,05beste     # simulated time between two steps in the simulation

numberOfRays = 1081          # spacing between two light rays (for distance calculation) in degrees
fov = 270                   # field of view in degree
timeFrames = 4              # number of past states used as an Input for the neural net

numbOfRobotsManual = 4     # only change if set to manual do not use more than 4
numbOfParallelEnvs = 15     # parallel environments are used to create more and diverse training experiences

scaleFactor = 65            # scales the simulation window (the window is also rezisable, only change if your display is low res)


levelFiles = ['Simple.svg', 'Funnel.svg']  #'SwapSide_a.svg'  fehlt mir


startTime = datetime.datetime.now().strftime("_%y-%m-%d--%H-%M")  # Timestamp used for saving the model

filename = "" # enter the filename from the models folder (without .h5 or .yml)
# filename = 'A2C_Network_2021-10-25--17-38_1400'
filename = 'A2C_Network_2021-11-22--00-12_200'

filename = 'PPO_21-12-01--18-38_e102'
filename = 'PPO_21-12-02--20-07_e74'
filename = 'PPO_21-12-09--17-16_e88'
filename = 'PPO_21-12-09--17-16_e29'
filename = 'PPO_21-12-14--10-26_e29'
filename = 'PPO_21-12-14--10-26_e90'
filename = 'PPO_21-12-14--12-06_e45'
filename = 'PPO_21-12-14--12-48_e103'

#TODO die arguments verschlanken
# Christian case zum standrt machen
# Case für Netz ohne yml File einabuen
# Netzparameter (lr, gamma in robins Netz verwenden, mit seinen Werten als default)
# entfernen: --alg  --shared --netSize  alle DQN-Parameter (arena width und length, noch verwendet?) --time_penalty --use_GPU
# nice to have:
# Timestamp in PPO_multi erzeugen und verwalten, nicht in args
# Wahrnehmungs_pipeline_training wieder lauffähig machen
# acivations wieder einbauen

#TODO in readme 2-3 Sätze zum Aufbau der svgs verlieren (wie muss man die bauen)
# kommentieren und praktikumsaufgabe updaten
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
    parser.add_argument('--gamma', type=float, default=gamma, help='Gamma')  # TODO ist gamma nur für DQN?
    parser.add_argument('--steps', type=int, default=steps, help='Steps in Environment per Episode')
    parser.add_argument('--train_interval', type=int, default=trainingInterval, help='The number of steps after which the neural net is trained.')
    parser.add_argument('--net_size', type=str, default='big', choices=['small', 'medium', 'big'], help='Determines the number of filters in the convolutional layers, the overall amount of neurons and the number of layers.')
    parser.add_argument('--shared', type=str2bool, default='True', help='Determines whether actor and aritic share their main network weights.')
    parser.add_argument('--load_christian', type=bool, default=True, help='Loads the best network ever trained by the master, be hold ... CHRISTIAN.')

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
    parser.add_argument('--number_of_rays', type=int, default=numberOfRays, help='The number of Rays emittet by the laser')
    parser.add_argument('--field_of_view', type=int, default=(fov/180 * math.pi), help='The lidars field of view in degree')
    parser.add_argument('--has_pie_slice',  type=str2bool, default='False', help='Determines if an Object is places on top of the robot to reflect pther robots lidar')
    parser.add_argument('--time_frames', type=int, default=timeFrames, help='Number of Timeframes which will be analyzed by neural net')
    parser.add_argument('--parallel_envs', type=int, default=numbOfParallelEnvs, help='Number of parallel environments used during training in addition to the main training process')
    parser.add_argument('--numb_of_robots', type=int, default=numbOfRobotsManual, help='Number of robots acting in one environment')
    parser.add_argument('--sim_time_step', type=float, default=simTimeStep, help='Time between steps')

    parser.add_argument('--training', type=bool, default=True, help='Training or Loading trained weights')
    parser.add_argument('--train_perception_only', type=bool, default=False, help='Training or Loading trained weights')
    parser.add_argument('--use_gpu', type=bool, default=False, help='Use GPUS with Tensorflow (Cuda 10.1 is needed)')
    parser.add_argument('--load_old', type=bool, default=True, help='Improve existing net (by loading pretrained weights and continuing with training)')
    parser.add_argument('--lidar_activation', type=bool, default=True, help='Show Lidar activation')

    parser.add_argument('--level_files', type=list, default=levelFiles, help='List of Level Files')


    args = parser.parse_args(args)

    if args.path == "":
        args.path = os.path.join(os.getcwd(), "models", "")

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    if args.manually:
        args.steps = 1000000
        args.parallel_envs = 1

    if (args.load_old or not args.training) and not args.load_christian:
        # Read YAML file
        with open(args.path + filename + ".yml", 'r') as stream:
            data_loaded = yaml.load(stream, Loader=yaml.UnsafeLoader)
            loadedArgs = data_loaded[0]
            args.steps = loadedArgs.steps
            args.time_frames = loadedArgs.time_frames
            args.time_penalty = loadedArgs.time_penalty
            args.number_of_rays = loadedArgs.number_of_rays
            args.field_of_view = loadedArgs.field_of_view
            args.has_pie_slice = loadedArgs.has_pie_slice
            args.net_size = loadedArgs.net_size
            args.shared = loadedArgs.shared
            args.load_christian = loadedArgs.load_christian
            #args.sim_time_step=loadedArgs.sim_time_step

    if args.load_christian:
        #args.load_old = True
        #filename = filenameChristian
        # args.steps = 600
        args.time_frames = 4
        args.time_penalty = False
        args.alg = 'ppo'
        args.number_of_rays = 1081
        args.field_of_view = 1.5*math.pi
        args.net_size = 'small'
        args.shared = False
        args.sim_time_step = 0.125#



    if args.mode == 'sonar':
        states = int((args.number_of_rays) + 7)

        #states = int((1081) + 7) #FÜR CHRISTIANS NETZ GEWICHTE
        env_dim = (args.time_frames, states)  # Timeframes, Robotstates

    elif args.mode == 'global':
        env_dim = (4, 9)


    # app = QApplication(sys.argv)
    # env = EnvironmentWithUI.Environment(app, args, env_dim[0], 0)


    act_dim = np.asarray(2)

    if(args.training):

        app = QApplication(sys.argv)
        if args.load_old:
            controlWindow = ControlWindow(app, args.parallel_envs, act_dim, env_dim, args, args.path+filename+'.h5')  # , model)
        else:
            controlWindow = ControlWindow(app, args.parallel_envs, act_dim, env_dim, args)  # , model)
        controlWindow.show()
        app.exec_()

    else:
        ray.init()
        model = None
        if args.alg == 'a2c':
            model = A2C_Multi(act_dim, env_dim, args)
            # model = PPO_Multi(act_dim, env_dim, args)
            # model = A2C(act_dim, env_dim, args)
        # elif args.alg == 'dqn':
        #     model = DQN(act_dim, env_dim, args)
        elif args.alg == 'ppo':
            model = PPO_Multi.remote(act_dim, env_dim, args)


        if args.train_perception_only:
            ray.get(model.load_net.remote(args.path + filename + '.h5'))
            ray.get(model.trainPerception.remote(args, env_dim[0]))
        else:
            ray.get(model.load_net.remote(args.path+filename+'.h5'))  # für model Laden .h5 auskommentieren
            ray.get(model.execute.remote(args, env_dim[0]))




# TODO Variable für Netzt bauen ja/ nein --> wenn ja dann den Typ --> falls netz bauen gewichte Laden? ---> wenn bauen Nein ganzes model laden
# TODO im Roboter nur goal oder station verwenden (für die position der Zielstation)

