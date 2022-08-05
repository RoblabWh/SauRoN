from PPO.SingleEnvironment import train
from Environment.Environment import Environment
from utils import str2bool

import sys
import torch
import argparse
from PyQt5.QtWidgets import QApplication
import numpy as np

training = True  # if training is set to false the trained model defined in the variable filename is loaded
levelFiles = ['tunnel.svg']
num_episodes = 150            # the number of epochs (/episodes) that are simulated
steps = 1200 #750            # number of steps per epoch (/episode)scaleFactor = 65            # scales the simulation window (the window is also rezisable, only change if your display is low res)
simTimeStep = 0.125          # simulation timestep (in seconds)
timeFrames = 4              # number of past states used as an Input for the neural net
numberOfRays = 1081
numbOfRobotsManual = 1      # only change if set to manual do not use more than 4

scaleFactor = 65            # scales the simulation window (the window is also rezisable, only change if your display is low res)
fov = 270                   # field of view in degree
manual = False              # if set to manual the number of robots is not changed

parser = argparse.ArgumentParser(description='PyTorch PPO for continuous controlling')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--env', type=str, default='BipedalWalker-v2', help='continuous env')
parser.add_argument('--render', default=False, action='store_true', help='Render?')
parser.add_argument('--solved_reward', type=float, default=55, help='stop training if avg_reward > solved_reward')
parser.add_argument('--print_interval', type=int, default=10, help='how many episodes to print the results out')
parser.add_argument('--save_interval', type=int, default=10, help='how many episodes to save a checkpoint')
parser.add_argument('--max_episodes', type=int, default=100000)
parser.add_argument('--max_timesteps', type=int, default=1500)
parser.add_argument('--update_timesteps', type=int, default=500, help='how many timesteps to update the policy')
parser.add_argument('--action_std', type=float, default=0.5, help='constant std for action distribution (Multivariate Normal)')
parser.add_argument('--K_epochs', type=int, default=80, help='update the policy for how long time everytime')
parser.add_argument('--eps_clip', type=float, default=0.2, help='epsilon for p/q clipped')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--seed', type=int, default=123, help='random seed to use')
parser.add_argument('--ckpt_folder', default='./models', help='Location to save checkpoint models')
parser.add_argument('--tb', default=False, action='store_true', help='Use tensorboardX?')
parser.add_argument('--log_folder', default='./logs', help='Location to save logs')
parser.add_argument('--mode', default='train', help='choose train or test')
parser.add_argument('--restore', default=False, action='store_true', help='Restore and go on training?')

parser.add_argument('--scale_factor', type=int, default=scaleFactor, help='Scale Factor for Environment')
parser.add_argument('--time_frames', type=int, default=timeFrames, help='Number of Timeframes (past States) which will be analyzed by neural net')
parser.add_argument('--manually', type=str2bool, nargs='?', const=True, default=manual,
                    help='Moving robot manually with wasd')
parser.add_argument('--training', type=str2bool, default=training, help='Training or Loading trained weights')
# Simulation settings
parser.add_argument('--level_files', type=list, default=levelFiles, help='List of level files as strings')
parser.add_argument('--numb_of_robots', type=int, default=numbOfRobotsManual,
                    help='Number of robots acting in one environment in the manual mode')
parser.add_argument('--nb_episodes', type=int, default=num_episodes, help='Number of training episodes')
parser.add_argument('--steps', type=int, default=steps, help='Steps in Environment per Episode')
parser.add_argument('--sim_time_step', type=float, default=simTimeStep, help='Time between steps')

# Robot settings
parser.add_argument('--number_of_rays', type=int, default=numberOfRays, help='The number of Rays emittet by the laser')
parser.add_argument('--field_of_view', type=int, default=fov, help='The lidars field of view in degree')
parser.add_argument('--has_pie_slice', type=str2bool, default='False',
                    help='Determines if an Object is places on top of the robot to reflect pther robots lidar')
parser.add_argument('--collide_other_targets', type=str2bool, default=False,
                    help='Determines whether the robot collides with targets of other robots (or passes through them)')  # Global oder Sonar einstellbar

parser.add_argument('--display_normals', type=bool, default=True,
                    help='Determines whether the normals of a wall are shown in the map.')
parser.add_argument('--train_perception_only', type=bool, default=False,
                    help='Improve existing net (works only if training is set to false')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args.field_of_view = args.field_of_view / 180 * np.pi

env_name = "tunnel"

states = int(1081 + 7)
env_dim = (4, states)  # Timeframes, Robotstates
app = QApplication(sys.argv)
env = Environment(app, args, env_dim[0], 0)

train(env_name, env,
      render=args.render, solved_reward=args.solved_reward,
      max_episodes=args.max_episodes, max_timesteps=args.max_timesteps, update_timestep=args.update_timesteps,
      action_std=args.action_std, K_epochs=args.K_epochs, eps_clip=args.eps_clip,
      gamma=args.gamma, lr=args.lr, betas=[0.9, 0.990], ckpt_folder=args.ckpt_folder,
      restore=args.restore, tb=args.tb, print_interval=args.print_interval, save_interval=args.save_interval)