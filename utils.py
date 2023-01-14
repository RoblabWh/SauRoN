from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import torch
from PIL import Image
import os
import time

import pickle


def initialize_output_weights(m, out_type):
    if out_type == 'actor':
        torch.nn.init.orthogonal_(m.weight.data, gain=0.01)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif out_type == 'critic':
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)

def initialize_hidden_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)

# normalizes a tensor to mean zero and standard deviation one
def normalize(tensor):
    return (tensor - tensor.mean()) / (tensor.std() + 1e-8)

def statesToObservationsTensor(list):
    states = np.asarray(list, dtype=object)
    laser = np.array(states[:, :, 0].tolist())
    ori = np.array(states[:, :, 1].tolist())
    dist = np.array(states[:, :, 2].tolist())
    vel = np.array(states[:, :, 3].tolist())
    return [torch.tensor(laser, dtype=torch.float32), torch.tensor(ori, dtype=torch.float32),
            torch.tensor(dist, dtype=torch.float32), torch.tensor(vel, dtype=torch.float32)]

def torchToNumpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
# TODO maybe use this ???!?!!
def _scan1DTo2D(lidarHits):

    #theta = np.radians(-135)
    #rotMatrix = np.array([[np.cos(theta), -np.sin(theta)],
    #                     [np.sin(theta), np.cos(theta)]])
    #data = np.dot(lidarHits, rotMatrix.T)
    data = lidarHits * 5
    data = data.astype(int)
    image = np.zeros((121, 121))

    # comment in to print scans in folder
    # image[data[:,0], data[:,1]] = 255
    #
    # im = Image.fromarray(image).convert('RGB')
    # frmt = "{0:06d}"
    # idx_ = len(os.listdir("./scans")) - 1
    # idx = frmt.format(idx_)
    # name = "./scans/" + idx + "_scan.png"
    # im.save(name)
    ######################################

    image[data[:, 0], data[:, 1]] = 1
    return image

def scan1DTo2D(distancesNorm, img_size, print=False):
    scanplot = []
    angle_min = 0
    angle_increment = np.radians(0.25)
    for i, point in enumerate(distancesNorm):
        angle = angle_min + (i * angle_increment)
        x = point * np.cos(angle)
        y = point * np.sin(angle)
        scanplot.append([x, y])
    scanplot = np.asarray(scanplot)
    theta = np.radians(-135)
    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    data = np.dot(scanplot, rotMatrix.T)
    data = ((data + 1) * int(img_size / 2)) - 1
    #data = ((data * 20) + 20) * 3
    data = data.astype(int)
    image = np.zeros((img_size, img_size))

    # prints scans in folder
    if print:
        image[data[:, 0], data[:, 1]] = 255
        im = Image.fromarray(image).convert('RGB')
        frmt = "{0:06d}"
        idx_ = len(os.listdir("./scans")) - 1
        idx = frmt.format(idx_)
        name = "./scans/" + idx + "_scan.png"
        im.save(name)

    image[data[:, 0], data[:, 1]] = 1
    return image

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# a wrapper function that computes computation time of a function
def timeit(f):
    def timed(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
              (f.__name__, args, kwargs, te-ts))
        return result
    return timed

# check if arguments are valid
def check_args(args):
    assert args.image_size > 0, "Image size must be positive"
    assert args.batches > 0, "Batches must be positive"
    assert args.lr > 0, "Learning rate must be positive"
    assert args.max_episodes > 0, "Number of episodes must be positive"
    assert args.time_frames > 0, "Number of time frames must be positive"
    assert args.print_interval > 0, "Print every must be positive"
    assert args.number_of_rays > 0, "Number of scans must be positive"
    assert args.update_experience > 0, "Update experience must be positive"
    assert args.update_experience > args.batches, "Update experience must be greater than batch size"
    assert args.visualization == "none" or args.visualization == "single" or args.visualization == "all", "Visualization must be none, single or all"
    assert os.path.exists(args.ckpt_folder), "Checkpoint folder does not exist."

class Logger(object):
    def __init__(self, log_dir, log_interval):
        self.writer = None
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.logging = False
        self.episode = 0
        self.last_logging_episode = 0
        # loss
        self.loss = []
        self.entropy = []
        self.critic_loss = []
        self.actor_loss = []

        self.actor_mean_linvel = []
        self.actor_mean_angvel = []
        self.actor_var_linvel = []
        self.actor_var_angvel = []

        self.reward = 0

        #objective
        self.objective_reached = 0
        self.number_of_agents = 0
        self.steps_agents = 0

    def __del__(self):
        if self.logging:
            self.close()

    def set_logging(self, logging):
        if logging:
            self.writer = SummaryWriter(self.log_dir)
        elif self.logging:
            self.close()
        self.logging = logging

    def build_graph(self, model, device):
        if self.logging:
            laser = torch.rand(4, 4, 1081).to(device)
            ori = torch.rand(4, 4, 2).to(device)
            dist = torch.rand(4, 4).to(device)
            vel = torch.rand(4, 4, 2).to(device)
            self.writer.add_graph(model, (laser, ori, dist, vel))

    def add_loss(self, loss, entropy, critic_loss, actor_loss):
        self.loss.append(loss)
        self.entropy.append(entropy)
        self.critic_loss.append(critic_loss)
        self.actor_loss.append(actor_loss)

    def summary_loss(self):
        if self.episode > self.last_logging_episode:
            if self.logging and not len(self.loss) == 0:
                self.writer.add_scalars('loss', {'loss': np.mean(self.loss),
                                                'entropy': np.mean(self.entropy),
                                                'critic_loss': np.mean(self.critic_loss),
                                                'actor loss': np.mean(self.actor_loss)}, self.episode)

    def add_step_agents(self, steps_agents):
        self.steps_agents += steps_agents

    def add_actor_output(self, actor_mean_linvel, actor_mean_angvel, actor_var_linvel, actor_var_angvel):
        self.actor_mean_linvel.append(actor_mean_linvel)
        self.actor_mean_angvel.append(actor_mean_angvel)
        self.actor_var_linvel.append(actor_var_linvel)
        self.actor_var_angvel.append(actor_var_angvel)

    def summary_actor_output(self):
        if self.logging and self.episode > self.last_logging_episode:
            self.writer.add_scalars('actor_output', {'Mean LinVel': np.mean(self.actor_mean_linvel),
                                                     'Mean AngVel': np.mean(self.actor_mean_angvel),
                                                     'Variance LinVel': np.mean(self.actor_var_linvel),
                                                     'Variance AngVel': np.mean(self.actor_var_angvel)}, self.episode)

    def summary_objective(self):
        if self.logging and self.episode > self.last_logging_episode:
            self.writer.add_scalar('objective reached', self.percentage_objective_reached(), self.episode)

    def add_reward(self, reward):
        self.reward += reward

    def percentage_objective_reached(self):
        return self.objective_reached / (self.episode - self.last_logging_episode)

    def add_objective(self, reachedGoals):
        self.objective_reached += (np.count_nonzero(reachedGoals) / self.number_of_agents)

    def set_number_of_agents(self, number_of_agents):
        self.number_of_agents = number_of_agents

    def summary_reward(self):
        if self.logging and self.episode > self.last_logging_episode:
            self.writer.add_scalar('reward per step', self.reward / self.steps_agents, self.episode)

    def summary_steps_agents(self):
        if self.logging and self.episode > self.last_logging_episode:
            self.writer.add_scalar('avg steps per agent', self.steps_agents / self.number_of_agents, self.episode)

    def log(self):

        self.summary_reward()
        objective_reached = self.percentage_objective_reached()
        self.summary_objective()
        self.summary_steps_agents()
        self.summary_actor_output()
        self.summary_loss()

        self.last_logging_episode = self.episode
        self.clear_summary()
        return self.reward, objective_reached

    def clear_summary(self):
        self.actor_mean_linvel = []
        self.actor_mean_angvel = []
        self.actor_var_linvel = []
        self.actor_var_angvel = []
        self.loss = []
        self.entropy = []
        self.critic_loss = []
        self.actor_loss = []
        self.objective_reached = 0
        self.steps_agents = 0
        self.reward = 0
        self.cnt_agents = 0
    def close(self):
        self.writer.close()
