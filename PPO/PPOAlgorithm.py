import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
import torch
import numpy as np
import random
import math
from utils import statesToTensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Inputspace(nn.Module):

    def __init__(self, scan_size):
        super(Inputspace, self).__init__()

        self.lidar_conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=16, stride=8)
        in_f = self.get_in_features(h_in=scan_size, kernel_size=16, stride=8)
        self.lidar_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=8, stride=4)
        in_f = self.get_in_features(h_in=in_f, kernel_size=8, stride=4)

        features_scan = int((in_f**2) * 32)
        self.flatten = nn.Flatten()
        self.lidar_flat = nn.Linear(in_features=128, out_features=160)
        self.concated_some = nn.Linear(in_features=180, out_features=96)

    def get_in_features(self, h_in, padding=0, dilation=1, kernel_size=0, stride=1):
        return (((h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)

    def forward(self, laser, orientation_to_goal, distance_to_goal, velocity):
        laser = F.relu(self.lidar_conv1(laser))
        laser = F.relu(self.lidar_conv2(laser))
        laser_flat = self.flatten(laser)

        laser_flat = self.lidar_flat(laser_flat)

        orientation_flat = self.flatten(orientation_to_goal)
        distance_flat = self.flatten(distance_to_goal)
        velocity_flat = self.flatten(velocity)
        concat = torch.cat([laser_flat, orientation_flat, distance_flat, velocity_flat], dim=1)
        densed = F.relu(self.concated_some(concat))

        return densed

class Actor(nn.Module):
    def __init__(self, scan_size):
        super(Actor, self).__init__()
        self.Inputspace = Inputspace(scan_size)

        # Mu
        self.mu = nn.Linear(in_features=96, out_features=2)
        # Var
        self.dense_var = nn.Linear(in_features=96, out_features=2)
        self.softplus = torch.nn.Softplus()

    def forward(self, laser, orientation_to_goal, distance_to_goal, velocity):
        x = self.Inputspace(laser.to(device), orientation_to_goal.to(device), distance_to_goal.to(device), velocity.to(device))
        mu = torch.tanh(self.mu(x))
        var = self.softplus(self.dense_var(x)) #torch.FloatTensor([0.0, 0.0])  # TODO:
        return [mu.to('cpu'), var.to('cpu')]

class Critic(nn.Module):
    def __init__(self, scan_size):
        super(Critic, self).__init__()
        self.Inputspace = Inputspace(scan_size)
        # Value
        self.value_temp = nn.Linear(out_features=128, in_features=96)
        self.value_temp2 = nn.Linear(out_features=128, in_features=96)
        self.value = nn.Linear(out_features=1, in_features=256, bias=False)

    def forward(self, laser, orientation_to_goal, distance_to_goal, velocity):
        x = self.Inputspace(laser, orientation_to_goal, distance_to_goal, velocity)
        value1 = F.relu(self.value_temp(x))
        value2 = F.relu(self.value_temp2(x))
        value_cat = torch.cat([value1, value2], dim=1)
        value = F.relu(self.value(value_cat))
        return value.to('cpu')

class ActorCritic(nn.Module):
    def __init__(self, action_std, scan_size):
        super(ActorCritic, self).__init__()
        action_dim = 2
        self.actor = Actor(scan_size)
        self.critic = Critic(scan_size)

        #self.action_var = torch.full((action_dim, ), action_std * action_std).to(device)

    def act(self, states, memory):
        laser, orientation, distance, velocity = states
        action_mean, action_std = self.actor(laser, orientation, distance, velocity)
        cov_mat = torch.diag_embed(action_std)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append([laser.to(device).detach(), orientation.to(device).detach(), distance.to(device).detach(), velocity.to(device).detach()])
        # memory.laser.append(laser[0].to(device).detach())
        # memory.orientation.append(orientation[0].to(device).detach())
        # memory.distance.append(distance[0].to(device).detach())
        # memory.velocity.append(velocity[0].to(device).detach())
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        laser, orientation, distance, velocity = zip(*state)
        laser = torch.stack(laser)
        orientation = torch.stack(orientation)
        distance = torch.stack(distance)
        velocity = torch.stack(velocity)
        state_value = self.critic(laser, orientation, distance, velocity)

        # to calculate action score(logprobs) and distribution entropy
        action_mean, action_std = self.actor(laser, orientation, distance, velocity)
        cov_mat = torch.diag_embed(action_std)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action.to('cpu'))
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, scan_size, action_std, lr, betas, gamma, K_epochs, eps_clip, restore=False, ckpt=None):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # current policy
        self.policy = ActorCritic(action_std, scan_size).to(device)
        if restore:
            pretained_model = torch.load(ckpt, map_location=lambda storage, loc: storage)
            self.policy.load_state_dict(pretained_model)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        # old policy: initialize old policy with current policy's parameter
        self.old_policy = ActorCritic(action_std, scan_size).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss()

    def select_action(self, states, memory):
        #state = torch.FloatTensor(state.reshape(1, -1)).to(device)  # flatten the state

        # prepare data
        states = statesToTensor(states)
        return self.old_policy.act(states, memory).cpu().numpy()

    def update(self, memory):
        # Monte Carlo estimation of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        # Normalize rewards
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = rewards.type(torch.float32)

        # convert list to tensor
        old_states = memory.states
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()

        # Train policy for K epochs: sampling and updating
        for _ in range(self.K_epochs):
            # Evaluate old actions and values using current policy
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Importance ratio: p/q
            ratios = torch.exp(logprobs - old_logprobs.to('cpu').detach())

            # Advantages
            advantages = rewards.to('cpu') - state_values.detach()

            # Actor loss using Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = - torch.min(surr1, surr2).type(torch.float32)

            # Critic loss: critic loss - entropy
            critic_loss = 0.5 * self.MSE_loss(rewards.to('cpu'), state_values) - 0.01 * dist_entropy

            # Total loss
            loss = actor_loss + critic_loss

            # Backward gradients
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights to old_policy
        self.old_policy.load_state_dict(self.policy.state_dict())