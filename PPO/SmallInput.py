import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import initialize_hidden_weights


class SmallInput(nn.Module):

    def __init__(self, scan_size):
        """
        A PyTorch Module that represents the input space of a neural network.

        This module takes in four inputs: a lidar scan, orientation to goal, distance to goal, and velocity.
        It then applies convolutional and dense layers to each input separately and concatenates the outputs
        to produce a flattened feature vector that can be fed into a downstream neural network.

        :param scan_size: The number of lidar scans in the input lidar scan.
        """
        super(SmallInput, self).__init__()

        self.time_steps = 4 #TODO make this dynamic
        self.flatten = nn.Flatten()
        self.out_features = 32

        layers_dict = [
            {'padding': 0, 'dilation': 1, 'kernel_size': 8, 'stride': 3, 'in_channels': self.time_steps, 'out_channels': 16},
            {'padding': 0, 'dilation': 1, 'kernel_size': 4, 'stride': 2, 'in_channels': 16, 'out_channels': 32},
            {'padding': 0, 'dilation': 1, 'kernel_size': 2, 'stride': 2, 'in_channels': 32, 'out_channels': 64},
        ]

        in_f = self.get_in_features(h_in=scan_size, layers_dict=layers_dict)
        laser_features = (int(in_f)) * layers_dict[-1]['out_channels']

        self.lidar_conv1 = nn.Conv1d(in_channels=layers_dict[0]['in_channels'], out_channels=layers_dict[0]['out_channels'],
                                        kernel_size=layers_dict[0]['kernel_size'], stride=layers_dict[0]['stride'],
                                        padding=layers_dict[0]['padding'], dilation=layers_dict[0]['dilation'])
        initialize_hidden_weights(self.lidar_conv1)
        self.lidar_conv2 = nn.Conv1d(in_channels=layers_dict[1]['in_channels'], out_channels=layers_dict[1]['out_channels'],
                                        kernel_size=layers_dict[1]['kernel_size'], stride=layers_dict[1]['stride'],
                                        padding=layers_dict[1]['padding'], dilation=layers_dict[1]['dilation'])
        initialize_hidden_weights(self.lidar_conv2)
        self.lidar_conv3 = nn.Conv1d(in_channels=layers_dict[2]['in_channels'], out_channels=layers_dict[2]['out_channels'],
                                        kernel_size=layers_dict[2]['kernel_size'], stride=layers_dict[2]['stride'],
                                        padding=layers_dict[2]['padding'], dilation=layers_dict[2]['dilation'])
        initialize_hidden_weights(self.lidar_conv3)

        ori_out_features = 8
        self.ori_dense = nn.Linear(in_features=2, out_features=ori_out_features)
        initialize_hidden_weights(self.ori_dense)

        dist_out_features = 4
        self.dist_dense = nn.Linear(in_features=1, out_features=dist_out_features)
        initialize_hidden_weights(self.dist_dense)

        vel_out_features = 4
        self.vel_dense = nn.Linear(in_features=2, out_features=vel_out_features)
        initialize_hidden_weights(self.vel_dense)

        lidar_out_features = 8
        self.lidar_flat = nn.Linear(in_features=laser_features, out_features=lidar_out_features)
        initialize_hidden_weights(self.lidar_flat)

        input_features = lidar_out_features + (ori_out_features + dist_out_features + vel_out_features) * self.time_steps

        self.input_dense = nn.Linear(in_features=input_features, out_features=64)
        initialize_hidden_weights(self.input_dense)
        self.input_dense2 = nn.Linear(in_features=64, out_features=self.out_features)
        initialize_hidden_weights(self.input_dense2)

    def get_in_features(self, h_in, layers_dict):
        for layer in layers_dict:
            padding = layer['padding']
            dilation = layer['dilation']
            kernel_size = layer['kernel_size']
            stride = layer['stride']
            h_in = ((h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
        return h_in

    def forward(self, laser, orientation_to_goal, distance_to_goal, velocity):
        laser = F.relu(self.lidar_conv1(laser))
        laser = F.relu(self.lidar_conv2(laser))
        laser = F.relu(self.lidar_conv3(laser))

        laser_flat = self.flatten(laser)

        orientation_to_goal = F.relu(self.ori_dense(orientation_to_goal))
        distance_to_goal = F.relu(self.dist_dense(distance_to_goal))
        velocity = F.relu(self.vel_dense(velocity))
        laser_flat = F.relu(self.lidar_flat(laser_flat))

        orientation_flat = self.flatten(orientation_to_goal)
        distance_flat = self.flatten(distance_to_goal)
        velocity_flat = self.flatten(velocity)

        concated_input = torch.cat((laser_flat, orientation_flat, distance_flat, velocity_flat), dim=1)
        input_dense = F.relu(self.input_dense(concated_input))
        input_dense = F.relu(self.input_dense2(input_dense))

        return input_dense