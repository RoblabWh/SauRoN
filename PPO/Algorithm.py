from utils import statesToObservationsTensor, normalize

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal
import torch
import numpy as np

from utils import initialize_hidden_weights, initialize_output_weights, torchToNumpy

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
            # self.lidar_conv1 = nn.Conv1d(in_channels=4, out_channels=12, kernel_size=7, stride=3)
            # initialize_hidden_weights(self.lidar_conv1)
            # in_f = self.get_in_features(h_in=scan_size, kernel_size=7, stride=3)
            # self.lidar_conv2 = nn.Conv1d(in_channels=12, out_channels=24, kernel_size=5, stride=2)
            # initialize_hidden_weights(self.lidar_conv2)
            # in_f = self.get_in_features(h_in=in_f, kernel_size=5, stride=2)
            # features_scan = (int(in_f)) * 24

            self.lidar_conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, stride=1)
            initialize_hidden_weights(self.lidar_conv1)
            in_f = self.get_in_features(h_in=scan_size, kernel_size=3, stride=1)
            self.lidar_conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
            initialize_hidden_weights(self.lidar_conv2)
            in_f = self.get_in_features(h_in=in_f, kernel_size=3, stride=1)
            self.maxPool = nn.MaxPool1d(kernel_size=2, stride=2)
            in_f = self.get_in_features(h_in=in_f, kernel_size=2, stride=2)
            self.lidar_conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
            initialize_hidden_weights(self.lidar_conv3)
            in_f = self.get_in_features(h_in=in_f, kernel_size=3, stride=1)
            features_scan = (int(in_f)) * 128

        self.flatten = nn.Flatten()

        ori_out_features = 32
        dist_out_features = 32
        vel_out_features = 32
        self.ori_dense = nn.Linear(in_features=2, out_features=ori_out_features)
        initialize_hidden_weights(self.ori_dense)
        self.dist_dense = nn.Linear(in_features=1, out_features=dist_out_features)
        initialize_hidden_weights(self.dist_dense)
        self.vel_dense = nn.Linear(in_features=2, out_features=vel_out_features)
        initialize_hidden_weights(self.vel_dense)

        # four is the number of timeframes TODO make this dynamic
        lidar_out_features = 192
        input_features = lidar_out_features + (ori_out_features + dist_out_features + vel_out_features) * 4

        self.lidar_flat = nn.Linear(in_features=features_scan, out_features=lidar_out_features)
        initialize_hidden_weights(self.lidar_flat)
        self.input_dense = nn.Linear(in_features=input_features, out_features=256)
        initialize_hidden_weights(self.input_dense)

    def get_in_features(self, h_in, padding=0, dilation=1, kernel_size=0, stride=1):
        return (((h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)

    def forward(self, laser, orientation_to_goal, distance_to_goal, velocity):
        laser = F.relu(self.lidar_conv1(laser))
        laser = F.relu(self.lidar_conv2(laser))
        laser = self.maxPool(laser)
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

        #other_dense = F.relu(self.other_flat(concat))
        #concat_all = torch.cat((laser_flat, other_dense), dim=1)
        #concat = torch.cat([laser_flat, orientation_flat, distance_flat, velocity_flat], dim=1)
        #densed = F.relu(self.concated_some(concat_all))

        return input_dense


class Actor(nn.Module):
    def __init__(self, scan_size, input_style):
        super(Actor, self).__init__()
        self.Inputspace = Inputspace(scan_size, input_style)

        # Mu
        self.mu = nn.Linear(in_features=256, out_features=2)
        initialize_output_weights(self.mu, 'actor')
        # Logstd
        self.log_std = nn.Parameter(torch.zeros(2, ))

    def forward(self, laser, orientation_to_goal, distance_to_goal, velocity):
        x = self.Inputspace(laser.to(device), orientation_to_goal.to(device), distance_to_goal.to(device), velocity.to(device))
        mu = torch.tanh(self.mu(x))
        std = torch.exp(self.log_std)
        var = torch.pow(std, 2)

        return mu.to('cpu'), var.to('cpu')


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
        return value


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

    def act(self, states):
        laser, orientation, distance, velocity = states
        # TODO: check if normalization of states is necessary
        # was suggested in: Implementation_Matters in Deep RL: A Case Study on PPO and TRPO
        action_mean, action_var = self.actor(laser, orientation, distance, velocity)

        cov_mat = torch.diag(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)
        ## logging of actions
        self.logger.add_actor_output(action_mean.mean(0)[0].item(), action_mean.mean(0)[1].item(), action_var[0].item(), action_var[1].item())

        action = dist.sample()
        action = torch.clip(action, -1, 1)
        action_logprob = dist.log_prob(action)

        return action, action_logprob

    def act_certain(self, states):
        laser, orientation, distance, velocity = states
        action, _ = self.actor(laser, orientation, distance, velocity)

        return action

    def evaluate(self, state, action):
        laser, orientation, distance, velocity = state
        state_value = self.critic(laser, orientation, distance, velocity)

        action_mean, action_var = self.actor(laser, orientation, distance, velocity)

        cov_mat = torch.diag(action_var.to(device))
        dist = MultivariateNormal(action_mean.to(device), cov_mat)
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

        self.optimizer_a = torch.optim.Adam(self.policy.actor.parameters(), lr=lr, betas=betas, eps=1e-5)
        self.optimizer_c = torch.optim.Adam(self.policy.critic.parameters(), lr=lr, betas=betas, eps=1e-5)

        # old policy: initialize old policy with current policy's parameter
        self.old_policy = ActorCritic(action_std, scan_size, input_style, logger).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss()

    def select_action(self, observations):
        # prepare data
        return self.old_policy.act(observations)

    def select_action_certain(self, observations):
        # prepare data
        return self.old_policy.act_certain(observations)

    def saveCurrentWeights(self, ckpt_folder, env_name):
        print('Saving current weights to ' + ckpt_folder + '/' + env_name + '_current.pth')
        torch.save(self.policy.state_dict(), ckpt_folder + '/PPO_continuous_{}_current.pth'.format(env_name))

    def get_advantages(self, values, masks, rewards):
        returns = []
        gae = 0
        a = [ai.unsqueeze(0) for ai in values]
        a.append(torch.tensor([0.], requires_grad=True).to(device))
        values = torch.cat(a).squeeze(0)
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * 0.95 * masks[i] * gae
            returns.insert(0, gae + values[i])

        returns = torch.FloatTensor(returns).to(device)
        adv = returns - values[:-1]
        return returns, (adv - adv.mean()) / (adv.std() + 1e-10)

    def update(self, memory, batches):
        # computes the discounted reward for every robots in memory
        rewards = []
        masks = []

        for robotmemory in memory.swarmMemory:
            _rewards = []
            _masks = []
            for reward, is_terminal in zip(reversed(robotmemory.rewards), reversed(robotmemory.is_terminals)):
                _masks.insert(0, 1 - is_terminal)
                _rewards.insert(0, reward)
            rewards.append(_rewards)
            masks.append(_masks)

        # flatten the rewards
        rewards = [item for sublist in rewards for item in sublist]
        masks = [item for sublist in masks for item in sublist]

        # Normalize rewards
        rewards = torch.tensor(rewards).type(torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = rewards.type(torch.float32)

        masks = torch.tensor(masks).to(device)

        # convert list to tensor
        old_states = memory.getObservationOfAllRobots()
        laser, orientation, distance, velocity = old_states
        old_actions = torch.stack(memory.getActionsOfAllRobots()).to(device)
        old_logprobs = torch.stack(memory.getLogProbsOfAllRobots()).to(device)

        # TODO randomize the order of experiences that it DOESNT interfer with GAE calculation
        # indices = torch.randperm(laser.shape[0])
        # laser = laser[indices]
        # orientation = orientation[indices]
        # distance = distance[indices]
        # velocity = velocity[indices]
        # old_actions = old_actions[indices]
        # old_logprobs = old_logprobs[indices]
        # rewards = rewards[indices]
        # masks = masks[indices]

        # Train policy for K epochs: sampling and updating
        for rewards_minibatch, old_laser_minibatch, old_orientation_minibatch, old_distance_minibatch, \
                old_velocity_minibatch, old_actions_minibatch, old_logprobs_minibatch, mask_minibatch in \
                zip(torch.tensor_split(rewards, batches), torch.tensor_split(laser, batches),
                    torch.tensor_split(orientation, batches), torch.tensor_split(distance, batches),
                    torch.tensor_split(velocity, batches), torch.tensor_split(old_actions, batches),
                    torch.tensor_split(old_logprobs, batches), torch.tensor_split(masks, batches)):

            old_states_minibatch = [old_laser_minibatch, old_orientation_minibatch, old_distance_minibatch,
                                    old_velocity_minibatch]
            _, values_, _ = self.policy.evaluate(old_states_minibatch, old_actions_minibatch)
            returns, advantages = self.get_advantages(values_.detach(), mask_minibatch, rewards_minibatch)

            for _ in range(self.K_epochs):
                # Evaluate old actions and values using current policy
                logprobs, values, dist_entropy = self.policy.evaluate(old_states_minibatch, old_actions_minibatch)

                # Importance ratio: p/q
                ratios = torch.exp(logprobs - old_logprobs_minibatch)

                # Advantages
                #returns, advantages = self.get_advantages(state_values.detach(), mask_minibatch, rewards_minibatch)
                #advantages = rewards_minibatch - state_values.detach()
                #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

                # Actor loss using Surrogate loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                entropy = 0.001 * dist_entropy
                actor_loss = - torch.min(surr1, surr2).type(torch.float32)

                # TODO CLIP VALUE LOSS ? Probably not necessary as according to:
                # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
                critic_loss_ = 0.5 * self.MSE_loss(returns, values)
                critic_loss = critic_loss_ - entropy

                # Total loss
                loss = actor_loss + critic_loss
                self.logger.add_loss(loss.detach().mean().item(), entropy=entropy.detach().mean().item(), critic_loss=critic_loss.detach().mean().item(), actor_loss=actor_loss.detach().mean().item())

                # Backward gradients
                #self.optimizer.zero_grad()
                self.optimizer_a.zero_grad()
                actor_loss.mean().backward(retain_graph=True)
                # Global gradient norm clipping https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=0.5)
                self.optimizer_a.step()

                self.optimizer_c.zero_grad()
                critic_loss.mean().backward()
                # Global gradient norm clipping https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html
                torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), max_norm=0.5)
                self.optimizer_c.step()
                #with torch.cuda.amp.autocast(True):
                #loss.mean().backward()


                # self.optimizer_a.step()
                # self.optimizer_c.step()
                #self.optimizer.step()

        # Copy new weights to old_policy
        self.old_policy.actor.load_state_dict(self.policy.actor.state_dict())
        self.old_policy.critic.load_state_dict(self.policy.critic.state_dict())
        #self.old_policy.load_state_dict(self.policy.state_dict())