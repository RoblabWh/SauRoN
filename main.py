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
from pathlib import Path
import warnings

####################################################################
######  Settings  you have to use/ change during this exercise  ####
####################################################################

manual = False   # manual lets you control a robot with w, a, s, d. (!!Maybe useful for testing rewards in combination with a print ;)  you should also lower the amount of robots during manual testing.)
filename = "A2C_Network_2021-11-22--00-12_200"    # enter the filename of the model file that you want to load (without .h5 or .yml, can be found in models folder)
training = True  # if training is set to false the trained model defined in the variable filename is loaded



# HYPERPARAMETERS

gamma = 0.999               # discount factor for calculating the discounted reward
lr = 0.0001                 # learning rate
num_episodes = 601          # the number of epochs (/episodes) that are simulated
steps = 750                 # number of steps per epoch (/episode)
trainingInterval = 75       # number of steps after which the neural net is trained
simTimeStep = 0.125         # simulated time between two steps in the simulation

numberOfRays = 1081         # spacing between two light rays (for distance calculation) in degrees
fov = 270                   # field of view in degree
timeFrames = 4              # number of past states used as an Input for the neural net

numbOfRobotsManual = 4     # only change if set to manual do not use more than 4
numbOfParallelEnvs = 12    # parallel environments are used to create more and diverse training experiences

scaleFactor = 65            # scales the simulation window (the window is also rezisable, only change if your display is low res)


levelFiles = ['Simple.svg', 'Funnel.svg', 'SwapSide_a.svg'] #, 'Lab.svg', 'Zipper.svg', 'svg2_tareq.svg', 'svg3_tareq.svg']


startTime = datetime.datetime.now().strftime("_%y-%m-%d--%H-%M")  # Timestamp used for saving the model



if __name__ == '__main__':
    args = None
    parser = argparse.ArgumentParser(description='Training parameters')
    # Main settings (Mode)
    parser.add_argument('--training', type=bool, default=training, help='Training or Loading trained weights')
    parser.add_argument('--train_perception_only', type=bool, default=False, help='Improve existing net (works only if training is set to false')
    parser.add_argument('--manually', type=str2bool, nargs='?', const=True, default=manual, help='Moving robot manually with wasd')
    parser.add_argument('--load_old', type=bool, default=False, help='Loading pre-trained weights/ models')
    parser.add_argument('--load_weights_only', type=bool, default=True, help='Checks whether a whole keras model is loaded or only weights for the configured network (may be problemetic in execute with show activations)')

    # Simulation settings
    parser.add_argument('--level_files', type=list, default=levelFiles, help='List of level files as strings')
    parser.add_argument('--numb_of_robots', type=int, default=numbOfRobotsManual, help='Number of robots acting in one environment in the manual mode')
    parser.add_argument('--nb_episodes', type=int, default=num_episodes, help='Number of training episodes')
    parser.add_argument('--steps', type=int, default=steps, help='Steps in Environment per Episode')
    parser.add_argument('--sim_time_step', type=float, default=simTimeStep, help='Time between steps')

    # Trainings settings
    parser.add_argument('--parallel_envs', type=int, default=numbOfParallelEnvs, help='Number of parallel environments used during training in addition to the main training process')
    parser.add_argument('--train_interval', type=int, default=trainingInterval, help='The number of steps after which the neural net is trained.')
    parser.add_argument('--save_intervall', type=int, default=50, help='Save Intervall')

    # PPO/ Network settings
    parser.add_argument('-lr', '--learningrate', type=float, default=lr, help='Learning Rate')
    parser.add_argument('--gamma', type=float, default=gamma, help='Gamma determines the influence of future rewards in a discounted reward')
    parser.add_argument('--time_frames', type=int, default=timeFrames, help='Number of Timeframes (past States) which will be analyzed by neural net')

    #System settings
    parser.add_argument('--path', type=str, default='', help='Path where Models are saved')
    parser.add_argument('--model_timestamp', type=str, default=startTime, help='Timestamp from when the model was created')
    parser.add_argument('--scale_factor', type=int, default=scaleFactor, help='Scale Factor for visualisation')
    parser.add_argument('--display_normals', type=bool, default=True, help='Determines whether the normals of a wall are shown in the map.')
    parser.add_argument('--lidar_activation', type=bool, default=True, help='Show Lidar activation')


    # Robot settings
    parser.add_argument('--number_of_rays', type=int, default=numberOfRays, help='The number of Rays emittet by the laser')
    parser.add_argument('--field_of_view', type=int, default=(fov/180 * math.pi), help='The lidars field of view in degree')
    parser.add_argument('--has_pie_slice',  type=str2bool, default='False', help='Determines if an Object is places on top of the robot to reflect pther robots lidar')
    parser.add_argument('--collide_other_targets', type=str2bool, default=False, help='Determines whether the robot collides with targets of other robots (or passes through them)')  # Global oder Sonar einstellbar

    # other settings
    parser.add_argument('--mode', type=str, default='sonar', choices=['global', 'sonar'], help='Training Mode')  #Global is deprecated and may not work


    args = parser.parse_args(args)

    if args.path == "":
        args.path = os.path.join(os.getcwd(), "models", "")

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    if args.manually:
        args.steps = 1000000
        args.parallel_envs = 1

    if (args.load_old or not args.training):
        filenameWithYml = filename + '.yml'
        pathToFile = Path(args.path + filenameWithYml)

        if pathToFile.exists():
            # Read YAML file
            with open(args.path + filename + '.yml', 'r') as stream:
                data_loaded = yaml.load(stream, Loader=yaml.UnsafeLoader)
                loadedArgs = data_loaded[0]
                args.steps = loadedArgs.steps
                args.time_frames = loadedArgs.time_frames
                args.number_of_rays = loadedArgs.number_of_rays
                args.field_of_view = loadedArgs.field_of_view
                args.has_pie_slice = loadedArgs.has_pie_slice
                args.sim_time_step = loadedArgs.sim_time_step
        else:
            warnings.warn("WARNING --> No YML File found, make sure that your configuration matches the trained model that you are trying to load! ")



    if args.mode == 'sonar':
        states = int((args.number_of_rays) + 7)

        env_dim = (args.time_frames, states)  # Timeframes, Robotstates

    elif args.mode == 'global':
        env_dim = (4, 9)

    act_dim = np.asarray(2)

    if(args.training):
        app = QApplication(sys.argv)
        if args.load_old:
            controlWindow = ControlWindow(app, args.parallel_envs, act_dim, env_dim, args, args.path+filename)  # , model)
        else:
            controlWindow = ControlWindow(app, args.parallel_envs, act_dim, env_dim, args)  # , model)
        controlWindow.show()
        app.exec_()

    else:
        ray.init()
        model = PPO_Multi.remote(act_dim, env_dim, args)

        if args.load_weights_only:
            filename += '.h5'
        if args.train_perception_only:
            ray.get(model.load_net.remote(args.path + filename))
            ray.get(model.trainPerception.remote(args, env_dim[0]))
        else:
            ray.get(model.load_net.remote(args.path+filename))
            ray.get(model.execute.remote(args, env_dim[0]))
