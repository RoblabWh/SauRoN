from utils import statesToTensor, normalize

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal
import torch
import numpy as np

from utils import initialize_hidden_weights, initialize_output_weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Inputspace(nn.Module):

    def __init__(self, scan_size, input_style):
        super(Inputspace, self).__init__()

        if input_style == 'image':
            self.lidar_conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=16, stride=4)
            initialize_hidden_weights(self.lidar_conv1)
            in_f = self.get_in_features(h_in=scan_size, kernel_size=16, stride=4)
            self.lidar_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
            initialize_hidden_weights(self.lidar_conv2)
            in_f = self.get_in_features(h_in=in_f, kernel_size=4, stride=2)
            features_scan = (int(in_f) ** 2) * 32
        else:
            self.lidar_conv1 = nn.Conv1d(in_channels=4, out_channels=12, kernel_size=7, stride=3)
            initialize_hidden_weights(self.lidar_conv1)
            in_f = self.get_in_features(h_in=scan_size, kernel_size=7, stride=3)
            self.lidar_conv2 = nn.Conv1d(in_channels=12, out_channels=24, kernel_size=5, stride=2)
            initialize_hidden_weights(self.lidar_conv2)
            in_f = self.get_in_features(h_in=in_f, kernel_size=5, stride=2)
            features_scan = (int(in_f)) * 24

        self.flatten = nn.Flatten()
        lidar_out_features = 192
        self.lidar_flat = nn.Linear(in_features=features_scan, out_features=lidar_out_features)
        initialize_hidden_weights(self.lidar_flat)
        self.other_flat = nn.Linear(in_features=20, out_features=64)
        initialize_hidden_weights(self.other_flat)
        self.concated_some = nn.Linear(in_features=lidar_out_features + 64, out_features=256)
        initialize_hidden_weights(self.concated_some)

    def get_in_features(self, h_in, padding=0, dilation=1, kernel_size=0, stride=1):
        return (((h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)

    def forward(self, laser, orientation_to_goal, distance_to_goal, velocity):
        laser = F.relu(self.lidar_conv1(laser))
        laser = F.relu(self.lidar_conv2(laser))
        laser_flat = self.flatten(laser)

        laser_flat = F.relu(self.lidar_flat(laser_flat))

        orientation_flat = self.flatten(orientation_to_goal)
        distance_flat = self.flatten(distance_to_goal)
        velocity_flat = self.flatten(velocity)
        concat = torch.cat((orientation_flat, distance_flat, velocity_flat), dim=1)
        other_dense = F.relu(self.other_flat(concat))
        densed = torch.cat((laser_flat, other_dense), dim=1)
        #concat = torch.cat([laser_flat, orientation_flat, distance_flat, velocity_flat], dim=1)
        #densed = F.relu(self.concated_some(concat_all))

        return densed


class Actor(nn.Module):
    def __init__(self, scan_size, input_style):
        super(Actor, self).__init__()
        self.Inputspace = Inputspace(scan_size, input_style)

        # Mu
        self.mu = nn.Linear(in_features=256, out_features=2)
        initialize_output_weights(self.mu, 'actor')
        # Var
        #logstds_param = nn.Parameter(torch.full((2,), -0.69))
        var_param = nn.Parameter(torch.full((2,), 1.0))
        self.register_parameter("var", var_param)

    def forward(self, laser, orientation_to_goal, distance_to_goal, velocity):
        x = self.Inputspace(laser.to(device), orientation_to_goal.to(device), distance_to_goal.to(device), velocity.to(device))
        mu = torch.tanh(self.mu(x))
        var = torch.clamp(self.var.exp(), 1e-2, 1)
        cov_mat = torch.diag(var)
        dist = MultivariateNormal(mu, cov_mat)

        return mu.to('cpu'), var.to('cpu'), dist


class Critic(nn.Module):
    def __init__(self, scan_size, input_style):
        super(Critic, self).__init__()
        self.Inputspace = Inputspace(scan_size, input_style)
        # Value
        self.value = nn.Linear(in_features=256, out_features=1)
        initialize_output_weights(self.value, 'critic')

    def forward(self, laser, orientation_to_goal, distance_to_goal, velocity):
        x = self.Inputspace(laser, orientation_to_goal, distance_to_goal, velocity)
        value = F.relu(self.value(x))
        return value.to('cpu')


class ActorCritic(nn.Module):
    def __init__(self, action_std, scan_size, input_style, logger):
        super(ActorCritic, self).__init__()
        action_dim = 2
        self.actor_cnt = 0
        self.critic_cnt = 0
        self.logger = logger
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(scan_size, input_style)
        self.critic = Critic(scan_size, input_style)

        # TODO statische var testen
        #self.logstds_param = nn.Parameter(torch.full((n_actions,), 0.1))
        #self.action_var = torch.full((action_dim, ), action_std * action_std).to(device)

    def act(self, states, memory):
        laser, orientation, distance, velocity = states
        # TODO: check if normalization of states is necessary
        # was suggested in: Implementation_Matters in Deep RL: A Case Study on PPO and TRPO
        action_mean, action_var, dist = self.actor(laser, orientation, distance, velocity)

        # logging of actions
        self.logger.add_actor_output(action_mean[0][0].item(), action_mean[0][1].item(), action_var[0].item(), action_var[1].item())

        action = dist.sample()
        action = torch.clip(action, -1, 1)
        action_logprob = dist.log_prob(action)

        memory.insertState(laser.to(device).detach(), orientation.to(device).detach(), distance.to(device).detach(), velocity.to(device).detach())
        memory.insertAction(action)
        memory.insertLogProb(action_logprob)

        return action.detach()

    def act_certain(self, states, memory):
        laser, orientation, distance, velocity = states
        action, _, _ = self.actor(laser, orientation, distance, velocity)

        memory.insertState(laser.to(device).detach(), orientation.to(device).detach(), distance.to(device).detach(), velocity.to(device).detach())
        memory.insertAction(action)

        return action.detach()

    def evaluate(self, state, action):
        laser, orientation, distance, velocity = state
        state_value = self.critic(laser, orientation, distance, velocity)

        _, _, dist = self.actor(laser, orientation, distance, velocity)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, scan_size, action_std, input_style, lr, betas, gamma, K_epochs, eps_clip, logger, restore=False, ckpt=None):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.logger = logger
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # current policy
        self.policy = ActorCritic(action_std, scan_size, input_style, logger).to(device)
        if restore:
            pretained_model = torch.load(ckpt, map_location=lambda storage, loc: storage)
            self.policy.load_state_dict(pretained_model)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas, eps=1e-5)

        # old policy: initialize old policy with current policy's parameter
        self.old_policy = ActorCritic(action_std, scan_size, input_style, logger).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss()

    def select_action(self, states, memory):
        # prepare data
        states = statesToTensor(states)
        return self.old_policy.act(states, memory).cpu().numpy()

    def select_action_certain(self, states, memory):
        # prepare data
        states = statesToTensor(states)
        return self.old_policy.act_certain(states, memory).cpu().numpy()

    def saveCurrentWeights(self, ckpt_folder, env_name):
        print('Saving current weights to ' + ckpt_folder + '/' + env_name + '_current.pth')
        torch.save(self.policy.state_dict(), ckpt_folder + '/PPO_continuous_{}.pth'.format(env_name))

    def update(self, memory, batch_size):
        # Monte Carlo estimation of rewards

        # TODO check if same ???!?!

        # rewards = []
        # discounted_reward = 0
        # for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
        #     if is_terminal:
        #         discounted_reward = 0
        #     discounted_reward = reward + self.gamma * discounted_reward
        #     rewards.insert(0, discounted_reward)

        # computes the discounted reward for every robots in memory
        rewards = []
        for robotmemory in memory.robotMemory:
            discounted_reward = 0
            _rewards = []
            for reward, is_terminal in zip(reversed(robotmemory.rewards), reversed(robotmemory.is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + self.gamma * discounted_reward
                _rewards.insert(0, discounted_reward)
                if is_terminal:
                    discounted_reward = 0
            rewards.append(_rewards)

        # flatten the rewards
        rewards = [item for sublist in rewards for item in sublist]

        # Normalize rewards
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = rewards.type(torch.float32)

        # convert list to tensor
        old_states = memory.getStatesOfAllRobots()
        laser, orientation, distance, velocity = old_states
        old_actions = torch.stack(memory.getActionsOfAllRobots()).to(device).detach()
        old_logprobs = torch.stack(memory.getLogProbsOfAllRobots()).to(device).detach()

        # Train policy for K epochs: sampling and updating
        for _ in range(self.K_epochs):
            for rewards_minibatch, old_laser_minibatch, old_orientation_minibatch, old_distance_minibatch, \
                old_velocity_minibatch, old_actions_minibatch, old_logprobs_minibatch in \
                zip(torch.tensor_split(rewards, batch_size), torch.tensor_split(laser, batch_size),
                    torch.tensor_split(orientation, batch_size), torch.tensor_split(distance, batch_size),
                    torch.tensor_split(velocity, batch_size), torch.tensor_split(old_actions, batch_size),
                    torch.tensor_split(old_logprobs, batch_size)):
                # Evaluate old actions and values using current policy
                old_states_minibatch = [old_laser_minibatch, old_orientation_minibatch, old_distance_minibatch, old_velocity_minibatch]
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_minibatch, old_actions_minibatch)

                # Importance ratio: p/q
                ratios = torch.exp(logprobs - old_logprobs_minibatch.detach()).to('cpu')

                # Advantages
                advantages = rewards_minibatch.to('cpu') - state_values.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

                # Actor loss using Surrogate loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                entropy = 0.001 * dist_entropy
                actor_loss = - torch.min(surr1, surr2).type(torch.float32)

                # TODO CLIP VALUE LOSS ? Probably not necessary as according to:
                # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
                critic_loss_ = 0.5 * self.MSE_loss(rewards_minibatch.to('cpu'), state_values)
                critic_loss = critic_loss_

                # Total loss
                loss = actor_loss + critic_loss - entropy.to('cpu')
                self.logger.add_loss(loss.mean().item(), entropy=entropy.mean().item(), critic_loss=critic_loss.mean().item(), actor_loss=actor_loss.mean().item())

                # Backward gradients
                self.optimizer.zero_grad()
                #with torch.cuda.amp.autocast(True):
                loss.mean().backward()
                # Global gradient norm clipping https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), max_norm=0.5)
                # actor_loss.mean().backward()
                # critic_loss.mean().backward()
                self.optimizer.step()

        # Copy new weights to old_policy
        self.old_policy.actor.load_state_dict(self.policy.actor.state_dict())
        self.old_policy.critic.load_state_dict(self.policy.critic.state_dict())
        #self.old_policy.load_state_dict(self.policy.state_dict())