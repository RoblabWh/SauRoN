from PPO.Environment import train, test
from Environment.Environment import Environment
from utils import str2bool, check_args
import random
import sys
import os
import argparse
from PyQt5.QtWidgets import QApplication
from types import SimpleNamespace

def startSimulation(args, level_files, createReward, get_advantages):

    # Train Parameters
    args['time_frames']=4
    args['max_episodes']=float('inf')
    args['action_std']=0.5
    args['eps_clip']=0.2
    args['input_style']='laser'
    args['image_size']=256

    # Simulation settings
    args['level_files']=level_files
    args['sim_time_step']=0.15

    # Robot settings
    args['number_of_rays']=1081
    args['field_of_view']=270
    args['has_pie_slice']=False
    args['collide_other_targets']=False
    args['manually']=False

    # Visualization & Managing settings
    args['visualization']="single"
    args['visualization_paused']=False
    args['tensorboard']=True
    args['print_interval']=1
    args['solved_percentage']=0.95
    args['log_interval']=50
    args['render']=False
    args['scale_factor']=55
    args['display_normals']=True

    args = SimpleNamespace(**args)

    if not os.path.exists(args.ckpt_folder):
        os.mkdir(args.ckpt_folder)
    check_args(args)

    app = QApplication(sys.argv)

    env = Environment(app, args, reward_func=createReward)

    if args.input_style == 'laser':
        args.image_size = args.number_of_rays

    if args.mode == 'train':
        train(args.model_name, env, inputspace=args.inputspace, solved_percentage=args.solved_percentage,
              max_episodes=args.max_episodes, max_timesteps=args.steps, update_experience=args.update_experience,
              _lambda=args._lambda, K_epochs=args.K_epochs, eps_clip=args.eps_clip,
              gamma=args.gamma, lr=args.lr, betas=[0.9, 0.990], ckpt_folder=args.ckpt_folder,
              restore=args.restore, log_interval=args.log_interval, scan_size=args.number_of_rays,
              batches=args.batches, tensorboard=args.tensorboard)
    elif args.mode == 'test':
        test(args.model_name, env, inputspace=args.inputspace,
            render=args.render, K_epochs=args.K_epochs, eps_clip=args.eps_clip,
            gamma=args.gamma, _lambda=args._lambda, lr=args.lr, betas=[0.9, 0.990], ckpt_folder=args.ckpt_folder, test_episodes=100,
            scan_size=args.image_size, advantages_func=get_advantages)
