from PPO.SingleEnvironment import train, test
from Environment.Environment import Environment
from utils import str2bool, check_args

import sys
import os
import argparse
from PyQt5.QtWidgets import QApplication

from mpi4py import MPI as mpi

mpi_rank = mpi.COMM_WORLD.Get_rank()
mpi_size = mpi.COMM_WORLD.Get_size()
print("mpi_size: ", mpi_size)

# use all svg files in the svg folder as default level_files
level_files = []
svg_path = os.path.join(os.path.split(sys.argv[0])[0], "svg")
for filename in os.listdir(svg_path):
    if os.path.isfile(os.path.join(svg_path, filename)):
        level_files.append(filename)
level_files.sort()

level_files = ['SimpleObstacles.svg', 'tunnel.svg', 'svg3_tareq.svg', 'engstelle.svg', 'Simple.svg', 'Funnel.svg']
level_files = ['onerobot.svg']
ckpt_folder = './models/test'
model_name = "model"

parser = argparse.ArgumentParser(description='SauRoN Simulation')
parser.add_argument('--ckpt_folder', default=ckpt_folder, help='Location to save checkpoint models')
parser.add_argument('--model_name', default=model_name, help='Name of the modelfile')
parser.add_argument('--mode', default='train', help='choose train or test')


# Train Parameters

parser.add_argument('--restore', default=False, action='store_true', help='Restore and go on training?')
parser.add_argument('--time_frames', type=int, default=4, help='Number of Timeframes (past States) which will be analyzed by neural net')
parser.add_argument('--steps', type=int, default=2000, help='Steps in Environment per Episode')
parser.add_argument('--max_episodes', type=int, default=10000000, help='Maximum Number of Episodes')
parser.add_argument('--update_experience', type=int, default=100000, help='how many experiences to update the policy')
parser.add_argument('--batch_size', type=int, default=5, help='batch size')
parser.add_argument('--action_std', type=float, default=0.5, help='constant std for action distribution (Multivariate Normal)') # TODO currently not used
parser.add_argument('--K_epochs', type=int, default=20, help='update the policy K times')
parser.add_argument('--eps_clip', type=float, default=0.2, help='epsilon for p/q clipped')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--input_style', default='laser', help='image or laser')
parser.add_argument('--image_size', type=float, default=256, help='size of the image that goes into the neural net')
parser.add_argument('--sync_experience', type=int, default=10000, help='how often to sync the experience')


# Simulation settings

parser.add_argument('--level_files', type=str, nargs='+', default=level_files, help='List of level files as strings')
parser.add_argument('--sim_time_step', type=float, default=0.125, help='Time between steps') #.125

# Robot settings

parser.add_argument('--number_of_rays', type=int, default=1081, help='The number of Rays emittet by the laser')
parser.add_argument('--field_of_view', type=int, default=270, help='The lidars field of view in degree')
parser.add_argument('--has_pie_slice', type=str2bool, default='False',
                    help='Determines if an Object is places on top of the robot to reflect other robots lidar')
parser.add_argument('--collide_other_targets', type=str2bool, default=False,
                    help='Determines whether the robot collides with targets of other robots (or passes through them)')
parser.add_argument('--manually', type=str2bool, nargs='?', const=True, default=False,
                    help='Moving robot manually with wasd')

# Visualization & Managing settings

parser.add_argument('--visualization', type=str, default="single", help="Visualization mode. none: Don't use any visualization; single: Show only the visualization of one process; all: Show all visualizations")
parser.add_argument('--tensorboard', type=str2bool, default=True, help='Use tensorboard')
parser.add_argument('--print_interval', type=int, default=1, help='how many episodes to print the results out')
parser.add_argument('--solved_percentage', type=float, default=0.99, help='stop training if objective is reached to this percentage')
parser.add_argument('--log_interval', type=int, default=5, help='how many episodes to log into tensorboard. Also regulates how solved percentage is calculated')
parser.add_argument('--render', default=False, action='store_true', help='Render?')
parser.add_argument('--scale_factor', type=int, default=55, help='Scale Factor for Environment')
parser.add_argument('--display_normals', type=bool, default=True,
                    help='Determines whether the normals of a wall are shown in the map.')
args = parser.parse_args()
check_args(args)

if mpi_rank == 0:
    print("Level files: ", args.level_files, flush=True)

level_index = mpi_rank % len(args.level_files)

app = None
if args.visualization == "single":
    if mpi_rank == 0:
        app = QApplication(sys.argv)
elif args.visualization == "all":
    app = QApplication(sys.argv)

env = Environment(app, args, args.time_frames, level_index)

# TODO schöner ???!!
if args.input_style == 'laser':
    args.image_size = args.number_of_rays

if args.mode == 'train':
    train(args.model_name, env, input_style=args.input_style, solved_percentage=args.solved_percentage,
          max_episodes=args.max_episodes, max_timesteps=args.steps, update_experience=args.update_experience,
          action_std=args.action_std, K_epochs=args.K_epochs, eps_clip=args.eps_clip,
          gamma=args.gamma, lr=args.lr, betas=[0.9, 0.990], ckpt_folder=args.ckpt_folder,
          restore=args.restore, log_interval=args.log_interval, scan_size=args.image_size,
          batch_size=args.batch_size, tensorboard=args.tensorboard, sync_experience=args.sync_experience)
elif args.mode == 'test':
    test(args.model_name, env, input_style=args.input_style,
         render=args.render, action_std=args.action_std, K_epochs=args.K_epochs, eps_clip=args.eps_clip,
         gamma=args.gamma, lr=args.lr, betas=[0.9, 0.990], ckpt_folder=args.ckpt_folder, test_episodes=100,
         scan_size=args.image_size)
