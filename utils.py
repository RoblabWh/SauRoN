from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import torch
from PIL import Image
import os
import time
from collections import deque
import pickle


def initialize_output_weights(m, out_type):
    """
    Initialize the weights of the output layer of the actor and critic networks
    :param m: the layer to initialize
    :param out_type: the type of the output layer (actor or critic)
    """
    if out_type == 'actor':
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif out_type == 'critic':
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)

def initialize_hidden_weights(m):
    """
    Initialize the weights of the hidden layers of the actor and critic networks
    :param m: the layer to initialize
    """
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)

def normalize(tensor):
    """
    Normalizes a tensor to mean zero and standard deviation one
    """
    return (tensor - tensor.mean()) / (tensor.std() + 1e-8)

def statesToObservationsTensor(list):
    """
    The observations are the laser scan, the orientation, the distance to the goal and the velocity.
    :param list: the list of states
    :return: a list of observations
    """
    states = np.asarray(list, dtype=object)
    laser = np.array(states[:, :, 0].tolist())
    ori = np.array(states[:, :, 1].tolist())
    dist = np.array(states[:, :, 2].tolist())
    vel = np.array(states[:, :, 3].tolist())
    if (laser.dtype or ori.dtype) == np.dtype("object"):
        print("warn")
    return [torch.tensor(laser, dtype=torch.float32), torch.tensor(ori, dtype=torch.float32),
            torch.tensor(dist, dtype=torch.float32), torch.tensor(vel, dtype=torch.float32)]

def torchToNumpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()

# TODO maybe use this ???!?!!
def _scan1DTo2D(lidarHits):
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
    """
    Converts a 1D scan to a 2D image
    :param distancesNorm: the 1D scan
    :param img_size: the size of the image
    :param print: if true, the image is saved in the scans folder
    """
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
    """
    Logger class for logging training and evaluation metrics. It uses tensorboardX to log the metrics.

    :param log_dir: (string) directory where the logs will be saved
    :param log_interval: (int) interval for logging
    """
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

        self.reward = {}

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

    def add_reward(self, rewards):
        for reward in rewards:
            for key in reward.keys():
                #quick and dirty change it
                if key in self.reward.keys():
                    self.reward[key] += reward[key]
                else:
                    self.reward[key] = reward[key]  

    def percentage_objective_reached(self):
        return self.objective_reached / (self.episode - self.last_logging_episode)

    def add_objective(self, reachedGoals):
        self.objective_reached += (np.count_nonzero(reachedGoals) / self.number_of_agents)

    def set_number_of_agents(self, number_of_agents):
        self.number_of_agents = number_of_agents

    def summary_reward(self):
        if self.logging and self.episode > self.last_logging_episode:
            self.reward['total'] = 0
            for key in self.reward.keys():
                if key != 'total':
                    reward_per_step = self.reward[key] / self.steps_agents
                    self.reward[key] = reward_per_step
                    self.reward['total'] += reward_per_step
            self.writer.add_scalars('reward', self.reward, self.episode)

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
        return sum([v for v in self.reward.values()]), objective_reached

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
        self.reward = {}
        self.cnt_agents = 0
    def close(self):
        self.writer.close()


class RunningMeanStd(object):
    """
    This class is used to calculate the running mean and standard deviation of a data.
    """
    # from https://github.com/openai/baselines
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.epsilon = 1e-8
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

    def get_std(self):
        return np.sqrt(self.var + self.epsilon)
    
class CircularBuffer:
    def __init__(self, size):
        self.buffer = [[-1, -1]] * size  # Initialize buffer with zeros
        self.index = 0  # Pointer to current position

    def add(self, x, y):
        self.buffer[self.index] = [x, y]  # Overwrite current position with new position
        self.index = (self.index + 1) % len(self.buffer)  # Move pointer to next position, wrap around if at end

    def count_invalid_positions(self):
        return self.buffer.count([-1, -1])

    def get_buffer(self):
        return self.buffer
    
def is_staying_in_place(buffer, threshold=1.0):
    # Make sure the buffer is full of valid positions
    if buffer.count_invalid_positions() > 0:
        return False

    all_positions = buffer.get_buffer()
    xs, ys = zip(*all_positions)  # Unpack coordinates

    return (max(xs) - min(xs)) < threshold and (max(ys) - min(ys)) < threshold


def distance(pos1, pos2):
    """
    Computes the Euclidean distance between two positions.

    Args:
        pos1 (tuple): The first position (x, y).
        pos2 (tuple): The second position (x, y).

    Returns:
        float: The Euclidean distance between the two positions.
    """
    return ((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)**0.5
