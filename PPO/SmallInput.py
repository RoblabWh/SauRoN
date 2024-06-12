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

        self.lidar_conv1 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, stride=1)
        initialize_hidden_weights(self.lidar_conv1)
        in_f = self.get_in_features(h_in=scan_size, kernel_size=3, stride=1)
        self.lidar_conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1)
        initialize_hidden_weights(self.lidar_conv2)
        in_f = self.get_in_features(h_in=in_f, kernel_size=3, stride=1)
        self.lidar_conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1)
        initialize_hidden_weights(self.lidar_conv3)
        in_f = self.get_in_features(h_in=in_f, kernel_size=3, stride=1)

        features_scan = (int(in_f)) * 16
        features_scan = 17200

        self.flatten = nn.Flatten()

        ori_out_features = 8
        dist_out_features = 8
        vel_out_features = 8

        self.ori_dense = nn.Linear(in_features=2, out_features=4)
        initialize_hidden_weights(self.ori_dense)
        self.ori_dense2 = nn.Linear(in_features=4, out_features=ori_out_features)
        initialize_hidden_weights(self.ori_dense2)

        self.dist_dense = nn.Linear(in_features=1, out_features=4)
        initialize_hidden_weights(self.dist_dense)
        self.dist_dense2 = nn.Linear(in_features=4, out_features=dist_out_features)
        initialize_hidden_weights(self.dist_dense2)

        self.vel_dense = nn.Linear(in_features=2, out_features=4)
        initialize_hidden_weights(self.vel_dense)
        self.vel_dense2 = nn.Linear(in_features=4, out_features=vel_out_features)
        initialize_hidden_weights(self.vel_dense2)

        # four is the number of timeframes TODO make this dynamic
        lidar_out_features = 32
        input_features = lidar_out_features + (ori_out_features + dist_out_features + vel_out_features) * 4

        self.lidar_flat = nn.Linear(in_features=features_scan, out_features=64)
        initialize_hidden_weights(self.lidar_flat)
        self.lidar_flat2 = nn.Linear(in_features=64, out_features=lidar_out_features)
        initialize_hidden_weights(self.lidar_flat2)

        self.input_dense = nn.Linear(in_features=input_features, out_features=256)
        initialize_hidden_weights(self.input_dense)
        self.input_dense2 = nn.Linear(in_features=256, out_features=128)
        initialize_hidden_weights(self.input_dense2)

    def get_in_features(self, h_in, padding=0, dilation=1, kernel_size=0, stride=1):
        return (((h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)

    def forward(self, laser, orientation_to_goal, distance_to_goal, velocity):
        laser = F.relu(self.lidar_conv1(laser))
        laser = F.relu(self.lidar_conv2(laser))
        laser = F.relu(self.lidar_conv3(laser))

        laser_flat = self.flatten(laser)

        orientation_to_goal = F.relu(self.ori_dense(orientation_to_goal))
        orientation_to_goal = F.relu(self.ori_dense2(orientation_to_goal))

        distance_to_goal = F.relu(self.dist_dense(distance_to_goal))
        distance_to_goal = F.relu(self.dist_dense2(distance_to_goal))

        velocity = F.relu(self.vel_dense(velocity))
        velocity = F.relu(self.vel_dense2(velocity))

        laser_flat = F.relu(self.lidar_flat(laser_flat))
        laser_flat = F.relu(self.lidar_flat2(laser_flat))

        orientation_flat = self.flatten(orientation_to_goal)
        distance_flat = self.flatten(distance_to_goal)
        velocity_flat = self.flatten(velocity)

        concated_input = torch.cat((laser_flat, orientation_flat, distance_flat, velocity_flat), dim=1)
        input_dense = F.relu(self.input_dense(concated_input))
        input_dense = F.relu(self.input_dense2(input_dense))

        return input_dense