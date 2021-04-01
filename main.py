import argparse
import numpy as np
import os
from PyQt5.QtWidgets import QApplication
import yaml

import EnvironmentWithUI
import sys
import datetime
from algorithms.DQN import DQN
from algorithms.A2C_parallel.A2C_Multi import A2C_Multi
from algorithms.utils import str2bool

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
steps = 1000

arenaWidth = 22   # m
arenaLength = 10  # m

scaleFactor = 69
angleStepsSonar = .5
timeFrames = 4
numbOfParallelEnvs = 4
numbOfRobots = 4

# taktischeZeit = datetime.datetime.now().strftime("%d%H%M%b%y")  # Zeitstempel beim Start des trainings für das gespeicherte Modell
startTime = datetime.datetime.now().strftime("_%y-%m-%d--%H-%M")  # Zeitstempel beim Start des trainings für das gespeicherte Modell



filename = 'A2C_actor_Critic_sonar_21-03-08--16-48_e565'
filename = 'A2C_21-03-11--11-24_e0'
filename = 'A2C_21-03-11--11-40_e89'
filename = 'A2C_21-03-11--14-24_e94'
filename = 'A2C_21-03-11--15-07'
filename = 'A2C_21-03-11--17-09_e634'
filename = 'A2C_21-03-26--21-46_e555'
filename = 'A2C_21-03-27--11-05'#_endOfLevel-4'
filename = 'A2C_21-03-27--15-35_endOfLevel-4'
filename = 'A2C_21-03-27--18-22'



if __name__ == '__main__':
    args = None
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--manually', type=str2bool, nargs='?', const=True, default=False, help='Moving robot manually with wasd')
    parser.add_argument('--nb_episodes', type=int, default=num_episodes, help='Number of training episodes')
    parser.add_argument('--save_intervall', type=int, default=50, help='Save Intervall')
    parser.add_argument('--path', type=str, default='', help='Path where Models are saved')
    parser.add_argument('--model_timestamp', type=str, default=startTime, help='Timestamp from when the model was created')
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

    parser.add_argument('--mode', type=str, default='sonar', choices=['global', 'sonar'], help='Training Mode')  # Global oder Sonar einstellbar
    parser.add_argument('--time_penalty', type=str2bool, default='False', help='Reward function with time step penalty')
    parser.add_argument('--angle_steps', type=int, default=angleStepsSonar, help='Angle Steps for sonar training')
    parser.add_argument('--time_frames', type=int, default=timeFrames, help='Number of Timeframes which will be analyzed by neural net')
    parser.add_argument('--parallel_envs', type=int, default=numbOfParallelEnvs, help='Number of parallel environments used during training in addition to the main training process')
    parser.add_argument('--numb_of_robots', type=int, default=numbOfRobots, help='Number of robots acting in one environment')

    parser.add_argument('--training', type=bool, default=True, help='Training or Loading trained weights')
    parser.add_argument('--load_old', type=bool, default=False, help='Improve existing net (by loading pretrained weights and continuing with training)')

    args = parser.parse_args(args)

    if args.path == "":
        args.path = os.path.join(os.getcwd(), "models", "")

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    if args.manually:
        args.steps = 1000000
        args.parallel_envs = 0

    if args.load_old or not args.training:
        # Read YAML file
        with open(args.path + filename + ".yml", 'r') as stream:
            data_loaded = yaml.load(stream, Loader=yaml.UnsafeLoader)
            loadedArgs =  data_loaded[0]
            args.steps = loadedArgs.steps
            args.time_frames = loadedArgs.time_frames
            args.time_penalty = loadedArgs.time_penalty
            args.angle_steps = loadedArgs.angle_steps



    if args.mode == 'sonar':
        states = int((360 / angleStepsSonar) + 7)
        env_dim = (args.time_frames, states)  # Timeframes, Robotstates

    elif args.mode == 'global':
        env_dim = (4, 9)


    app = QApplication(sys.argv)
    env = EnvironmentWithUI.Environment(app, args, env_dim[0], 3)


    act_dim = np.asarray(env.get_actions()) #TODO bei kontinuierlichem 2 actions

    if args.alg == 'a2c':
        model = A2C_Multi(act_dim, env_dim, args)
        # model = A2C(act_dim, env_dim, args)
    elif args.alg == 'dqn':
        model = DQN(act_dim, env_dim, args)

    if args.training:
        if args.load_old:
            model.load_weights(args.path+filename+'.h5')
        model.train(env)
    elif not args.training:
        model.load_weights(args.path+filename+'.h5')
        model.execute(env, args)





