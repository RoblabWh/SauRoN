from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import torch
from PIL import Image
import os
import time

from mpi4py import MPI
import pickle

MPI_SEND_LIMIT = 2**31

def mpi_send(obj, comm:MPI.Comm, dest:int):
    data = pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
    comm.send(len(data), dest)
    while len(data) > 0:
        comm.Send(data[:MPI_SEND_LIMIT], dest)
        data = data[MPI_SEND_LIMIT:]

def mpi_recv(comm:MPI.Comm, source:int = MPI.ANY_SOURCE):
    data = bytes()
    status = MPI.Status()
    datalen = comm.recv(status=status)
    source = status.Get_source()
    recv_buffer = bytearray(min(datalen, MPI_SEND_LIMIT))
    while len(data) < datalen:
        comm.Recv(recv_buffer, source=source, status=status)
        data += recv_buffer[:status.Get_count()]
    return pickle.loads(data)

def mpi_reduce(obj, comm:MPI.Comm):
    if comm.Get_rank() == 0:
        for i in range(1, comm.Get_size()):
            obj += mpi_recv(comm, i)
        return obj
    else:
        mpi_send(obj, comm, 0)

def mpi_comm_split(comm:MPI.Comm, color:int=0, key:int=0):
    assert comm != MPI.COMM_WORLD

    new_comm = comm.Split(color, key)
    comm.Disconnect()
    return new_comm

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

def statesToTensor(list):
    states = np.asarray(list, dtype=object)
    laser = np.array(states[:, :, 0].tolist())
    ori = np.array(states[:, :, 1].tolist())
    dist = np.array(states[:, :, 2].tolist())
    vel = np.array(states[:, :, 3].tolist())
    return [torch.tensor(laser, dtype=torch.float32), torch.tensor(ori, dtype=torch.float32),
            torch.tensor(dist, dtype=torch.float32), torch.tensor(vel, dtype=torch.float32)]

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
    assert args.batch_size > 0, "Batch size must be positive"
    assert args.lr > 0, "Learning rate must be positive"
    assert args.max_episodes > 0, "Number of episodes must be positive"
    assert args.time_frames > 0, "Number of time frames must be positive"
    assert args.print_interval > 0, "Print every must be positive"
    assert args.number_of_rays > 0, "Number of scans must be positive"
    assert args.update_experience > 0, "Update experience must be positive"
    assert args.update_experience > args.batch_size, "Update experience must be greater than batch size"
    assert args.visualization == "none" or args.visualization == "single" or args.visualization == "all", "Visualization must be none, single or all"
    assert os.path.exists(args.ckpt_folder), "Checkpoint folder does not exist."

class Logger(object):
    def __init__(self, log_dir):
        self.writer = None
        self.log_dir = log_dir
        self.logging = False
        self.episode = 0
        self.last_episode = 0
        # loss
        self.loss = []
        self.entropy = []
        self.critic_loss = []
        self.actor_loss = []
        self.temp_loss = []
        self.temp_entropy = []
        self.temp_critic_loss = []
        self.temp_actor_loss = []

        self.actor_mean_linvel = []
        self.actor_mean_angvel = []
        self.actor_var_linvel = []
        self.actor_var_angvel = []
        self.temp_actor_mean_linvel = []
        self.temp_actor_mean_angvel = []
        self.temp_actor_var_linvel = []
        self.temp_actor_var_angvel = []

        self.steps = 0
        self.temp_steps = 0
        self.reward = 0
        self.temp_reward = 0

        #objective
        self.objective_reached = 0
        self.temp_objective_reached = 0
        self.number_of_agents = 0

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
        self.temp_loss.append(loss)
        self.temp_entropy.append(entropy)
        self.temp_critic_loss.append(critic_loss)
        self.temp_actor_loss.append(actor_loss)

    def summary_loss(self, timepoint):
        if self.episode > self.last_episode:
            if self.logging and not len(self.loss) == 0:
                self.writer.add_scalars('loss', {'loss': np.mean(self.loss),
                                                'entropy': np.mean(self.entropy),
                                                'critic_loss': np.mean(self.critic_loss),
                                                'actor loss': np.mean(self.actor_loss)}, timepoint)
            self.loss = []
            self.entropy = []
            self.critic_loss = []
            self.actor_loss = []

    def add_actor_output(self, actor_mean_linvel, actor_mean_angvel, actor_var_linvel, actor_var_angvel):
        self.temp_actor_mean_linvel.append(actor_mean_linvel)
        self.temp_actor_mean_angvel.append(actor_mean_angvel)
        self.temp_actor_var_linvel.append(actor_var_linvel)
        self.temp_actor_var_angvel.append(actor_var_angvel)

    def summary_actor_output(self, actor_mean_linvel, actor_mean_angvel, actor_var_linvel, actor_var_angvel, timepoint):
        if self.logging and self.episode > self.last_episode:
            self.writer.add_scalars('actor_output', {'Mean LinVel': actor_mean_linvel,
                                                     'Mean AngVel': actor_mean_angvel,
                                                     'Variance LinVel': actor_var_linvel,
                                                     'Variance AngVel': actor_var_angvel}, timepoint)

    def summary_objective(self, objective_reached, timepoint):
        if self.logging and self.episode > self.last_episode:
            self.writer.add_scalar('percentage_objective_reached', objective_reached, timepoint)

    def add_reward(self, reward):
        self.temp_reward += reward

    def percentage_objective_reached(self):
        return self.objective_reached / ((self.episode - self.last_episode) * self.number_of_agents)

    def add_objective(self, reachedGoals):
        self.temp_objective_reached += np.count_nonzero(reachedGoals)

    def log_episode(self, episode):
        self.episode = episode

        self.loss += self.temp_loss
        self.entropy += self.temp_entropy
        self.critic_loss += self.temp_critic_loss
        self.actor_loss += self.temp_actor_loss
        self.temp_loss = []
        self.temp_entropy = []
        self.temp_critic_loss = []
        self.temp_actor_loss = []

        self.actor_mean_linvel += self.temp_actor_mean_linvel
        self.actor_mean_angvel += self.temp_actor_mean_angvel
        self.actor_var_linvel += self.temp_actor_var_linvel
        self.actor_var_angvel += self.temp_actor_var_angvel
        self.temp_actor_mean_linvel = []
        self.temp_actor_mean_angvel = []
        self.temp_actor_var_linvel = []
        self.temp_actor_var_angvel = []

        self.steps += self.temp_steps
        self.temp_steps = 0
        self.reward += self.temp_reward
        self.temp_reward = 0

        self.objective_reached += self.temp_objective_reached
        self.temp_objective_reached = 0

    def set_number_of_agents(self, number_of_agents):
        self.number_of_agents = number_of_agents

    def add_steps(self, steps):
        self.temp_steps += steps

    def summary_reward(self, reward, timepoint):
        if self.logging and self.episode > self.last_episode:
            self.writer.add_scalar('reward', reward, timepoint)

    def summary_steps(self, steps, timepoint):
        if self.logging and self.episode > self.last_episode:
            self.writer.add_scalar('steps', steps, timepoint)

    def get_means(self):
        if self.episode > self.last_episode:
            episode_count = self.episode - self.last_episode
            return [True,\
                   self.reward / episode_count,\
                   self.percentage_objective_reached(),\
                   self.steps / episode_count,\
                   np.mean(self.actor_mean_linvel),\
                   np.mean(self.actor_mean_angvel),\
                   np.mean(self.actor_var_linvel),\
                   np.mean(self.actor_var_angvel)]
        else:
            return [False, 0, 0, 0, 0, 0, 0, 0]

    def clear_summary(self):
        if self.episode > self.last_episode:
            self.actor_mean_linvel = []
            self.actor_mean_angvel = []
            self.actor_var_linvel = []
            self.actor_var_angvel = []
            self.objective_reached = 0
            self.steps = 0
            self.reward = 0
            self.last_episode = self.episode

    def close(self):
        self.writer.close()
