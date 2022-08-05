from PPO.SingleEnvironment import train, test
from Environment.Environment import Environment
from utils import str2bool

import sys
import torch
import argparse
from PyQt5.QtWidgets import QApplication

levelFiles = ['tunnel.svg']
env_name = "tunnel"

parser = argparse.ArgumentParser(description='PyTorch PPO for continuous controlling')
parser.add_argument('--save_interval', type=int, default=100, help='how many episodes to save a checkpoint')
parser.add_argument('--ckpt_folder', default='./models', help='Location to save checkpoint models')
parser.add_argument('--mode', default='train', help='choose train or test')


# Train Parameters

parser.add_argument('--restore', default=False, action='store_true', help='Restore and go on training?')
parser.add_argument('--time_frames', type=int, default=4, help='Number of Timeframes (past States) which will be analyzed by neural net')
parser.add_argument('--steps', type=int, default=1500, help='Steps in Environment per Episode')
parser.add_argument('--max_episodes', type=int, default=100000)
parser.add_argument('--update_timesteps', type=int, default=120, help='how many timesteps to update the policy')
parser.add_argument('--action_std', type=float, default=0.5, help='constant std for action distribution (Multivariate Normal)')
parser.add_argument('--K_epochs', type=int, default=5, help='update the policy for how long time everytime')
parser.add_argument('--eps_clip', type=float, default=0.2, help='epsilon for p/q clipped')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--solved_reward', type=float, default=55, help='stop training if avg_reward > solved_reward')
parser.add_argument('--image_size', type=float, default=256, help='size of the image that goes into the neural net')


# Simulation settings

parser.add_argument('--level_files', type=list, default=levelFiles, help='List of level files as strings')
parser.add_argument('--sim_time_step', type=float, default=0.125, help='Time between steps')
parser.add_argument('--numb_of_robots', type=int, default=1,
                    help='Number of robots acting in one environment in the manual mode')

# Robot settings

parser.add_argument('--number_of_rays', type=int, default=1081, help='The number of Rays emittet by the laser')
parser.add_argument('--field_of_view', type=int, default=270, help='The lidars field of view in degree')
parser.add_argument('--has_pie_slice', type=str2bool, default='False',
                    help='Determines if an Object is places on top of the robot to reflect pther robots lidar')
parser.add_argument('--collide_other_targets', type=str2bool, default=False,
                    help='Determines whether the robot collides with targets of other robots (or passes through them)')  # Global oder Sonar einstellbar
parser.add_argument('--manually', type=str2bool, nargs='?', const=True, default=False,
                    help='Moving robot manually with wasd')

# Visualization settings

parser.add_argument('--print_interval', type=int, default=10, help='how many episodes to print the results out')
parser.add_argument('--render', default=False, action='store_true', help='Render?')
parser.add_argument('--scale_factor', type=int, default=65, help='Scale Factor for Environment')
parser.add_argument('--display_normals', type=bool, default=True,
                    help='Determines whether the normals of a wall are shown in the map.')
args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
app = QApplication(sys.argv)
env = Environment(app, args, args.time_frames, 0)

if args.mode == 'train':
    train(env_name, env,
          render=args.render, solved_reward=args.solved_reward,
          max_episodes=args.max_episodes, max_timesteps=args.steps, update_timestep=args.update_timesteps,
          action_std=args.action_std, K_epochs=args.K_epochs, eps_clip=args.eps_clip,
          gamma=args.gamma, lr=args.lr, betas=[0.9, 0.990], ckpt_folder=args.ckpt_folder,
          restore=args.restore, print_interval=args.print_interval, save_interval=args.save_interval, scan_size=args.image_size)
elif args.mode == 'test':
    test(env_name, env,
         render=args.render, action_std=args.action_std, K_epochs=args.K_epochs, eps_clip=args.eps_clip,
         gamma=args.gamma, lr=args.lr, betas=[0.9, 0.990], ckpt_folder=args.ckpt_folder, test_episodes=100, scan_size=args.image_size)