from utils import statesToObservationsTensor, normalize

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal
import torch
import os, warnings
import numpy as np
from pathlib import Path
from utils import initialize_hidden_weights, initialize_output_weights, torchToNumpy, RunningMeanStd
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Inputspace(nn.Module):

    def __init__(self, scan_size, input_style):
        """
        A PyTorch Module that represents the input space of a neural network.

        This module takes in four inputs: a lidar scan, orientation to goal, distance to goal, and velocity.
        It then applies convolutional and dense layers to each input separately and concatenates the outputs
        to produce a flattened feature vector that can be fed into a downstream neural network.

        :param scan_size: The number of lidar scans in the input lidar scan.
        """
        super(Inputspace, self).__init__()

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

        self.input_dense = nn.Linear(in_features=input_features, out_features=128)
        initialize_hidden_weights(self.input_dense)
        self.input_dense2 = nn.Linear(in_features=128, out_features=256)
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



class Actor(nn.Module):
    """
    A PyTorch Module that represents the actor network of a PPO agent.

    This module takes in four inputs: a lidar scan, orientation to goal, distance to goal, and velocity.
    It then applies convolutional and dense layers to each input separately and concatenates the outputs
    to produce a flattened feature vector that can be fed into a downstream neural network.

    :param scan_size: The number of lidar scans in the input lidar scan.
    """
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
    """
    A PyTorch Module that represents the critic network of a PPO agent.

    This module takes in four inputs: a lidar scan, orientation to goal, distance to goal, and velocity.
    It then applies convolutional and dense layers to each input separately and concatenates the outputs
    to produce a flattened feature vector that can be fed into a downstream neural network.

    :param scan_size: The number of lidar scans in the input lidar scan.
    """
    def __init__(self, scan_size, input_style):
        super(Critic, self).__init__()
        self.Inputspace = Inputspace(scan_size, input_style)
        # Value
        self.value = nn.Linear(in_features=256, out_features=1)
        initialize_output_weights(self.value, 'critic')

    def forward(self, laser, orientation_to_goal, distance_to_goal, velocity):
        x = self.Inputspace(laser, orientation_to_goal, distance_to_goal, velocity)
        value = self.value(x)
        return value


class ActorCritic(nn.Module):
    """
    A PyTorch Module that represents the actor-critic network of a PPO agent.

    This module takes in four inputs: a lidar scan, orientation to goal, distance to goal, and velocity.
    It then applies convolutional and dense layers to each input separately and concatenates the outputs
    to produce a flattened feature vector that can be fed into a downstream neural network.
    """
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
        """
        Returns an action sampled from the actor's distribution and the log probability of that action.

        :param states: A tuple of the current lidar scan, orientation to goal, distance to goal, and velocity.
        :return: A tuple of the sampled action and the log probability of that action.
        """
        with torch.no_grad():
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
        """
        Returns an action from the actor's distribution without sampling.

        :param states: A tuple of the current lidar scan, orientation to goal, distance to goal, and velocity.
        :return: The action from the actor's distribution.
        """
        with torch.no_grad():
            laser, orientation, distance, velocity = states
            action, _ = self.actor(laser, orientation, distance, velocity)

        return action

    def evaluate(self, state, action):
        """
        Returns the log probability of the given action, the value of the given state, and the entropy of the actor's
        distribution.

        :param state: A tuple of the current lidar scan, orientation to goal, distance to goal, and velocity.
        :param action: The action to evaluate.
        :return: A tuple of the log probability of the given action, the value of the given state, and the entropy of the
        actor's distribution.
        """
        laser, orientation, distance, velocity = state
        state_value = self.critic(laser, orientation, distance, velocity)

        action_mean, action_var = self.actor(laser, orientation, distance, velocity)

        cov_mat = torch.diag(action_var.to(device))
        dist = MultivariateNormal(action_mean.to(device), cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    """
    This class represents the PPO Algorithm. It is used to train an actor-critic network.

    :param scan_size: The number of lidar scans in the input lidar scan.
    :param action_std: The standard deviation of the action distribution.
    :param input_style: The style of the input to the network.
    :param lr: The learning rate of the network.
    :param betas: The betas of the Adam optimizer.
    :param gamma: The discount factor.
    :param K_epochs: The number of epochs to train the network.
    :param eps_clip: The epsilon value for clipping.
    :param logger: The logger to log data to.
    :param restore: Whether to restore the network from a checkpoint.
    :param ckpt: The checkpoint to restore from.
    """

    def __init__(self, scan_size, action_std, input_style, lr, betas, gamma, _lambda, K_epochs, eps_clip, logger, restore=False, ckpt=None, advantages_func=None):
        # Algorithm parameters
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self._lambda = _lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # Folder the models are stored in
        if not restore:
            if os.path.isfile(ckpt):
                raise ValueError("Model path must be a directory, not a file")
            if not os.path.exists(ckpt):
                print(f"Creating model directory under {os.path.abspath(ckpt)}")
                os.makedirs(ckpt)
        self.model_path = Path(ckpt)

        # Current Policy
        self.policy = ActorCritic(action_std, scan_size, input_style, logger).to(device)
        if restore:
            self.load_model(Path(ckpt))

        self.optimizer_a = torch.optim.Adam(self.policy.actor.parameters(), lr=lr, betas=betas, eps=1e-5)
        self.optimizer_c = torch.optim.Adam(self.policy.critic.parameters(), lr=lr, betas=betas, eps=1e-5)

        self.MSE_loss = nn.MSELoss()
        self.running_reward_std = RunningMeanStd()
        if advantages_func is not None:
            self.advantage_func = advantages_func
        else:
            self.advantage_func = self.get_advantages


    def set_eval(self):
        self.policy.eval()

    def load_model(self, path):
        try:
            self.policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
            return True
        except FileNotFoundError:
            warnings.warn(f"Could not restore model from {path}. Falling back to train mode.")
            return False

    def select_action(self, observations):
        return self.policy.act(observations)

    def select_action_certain(self, observations):
        return self.policy.act_certain(observations)

    def saveCurrentWeights(self, name):
        print('Saving current weights to ' + str(self.model_path.parent) + '/PPO_continuous_{}.pth'.format(name))
        torch.save(self.policy.state_dict(), str(self.model_path.parent) + '/PPO_continuous_{}.pth'.format(name))

    def calculate_returns(self, rewards, normalize=False):

        returns = []
        return_ = 0

        for r in reversed(rewards):
            return_ = r + return_ * self.gamma
            returns.insert(0, return_)

        returns = torch.tensor(returns, dtype=torch.float32)

        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns.detach().to(device)

    def get_advantages(self, gamma, _lambda, values, masks, rewards):
        """
        Computes the advantages of the given rewards and values.

        :param values: The values of the states.
        :param masks: The masks of the states.
        :param rewards: The rewards of the states.
        :return: The advantages of the states.
        """
        advantages = []
        returns = []
        gae = 0

        for i in reversed(range(len(rewards))):
            delta = rewards[i] - values[i]
            if masks[i] == 0:
                delta = delta + gamma * values[i + 1]
            gae = delta + gamma * _lambda * masks[i] * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        advantages = torch.FloatTensor(advantages).to(device)
        if advantages.numel() > 1:
            norm_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        else:
            norm_adv = advantages / (torch.abs(advantages) + 1e-10)

        returns = torch.FloatTensor(returns).to(device)
        # norm_returns = (returns - returns.mean()) / (returns.std() + 1e-10)

        return norm_adv, returns

    def get_advantages_returns(self, states, actions, masks, rewards):
        # Advantages
        with torch.no_grad():
            _, values_, _ = self.policy.evaluate(states, actions)
            if values_.dim() == 0:
                values_ = values_.unsqueeze(0)
            if masks[-1] == 0:
                last_state = (states[0][-1].unsqueeze(0),
                              states[1][-1].unsqueeze(0),
                              states[2][-1].unsqueeze(0),
                              states[3][-1].unsqueeze(0))
                bootstrapped_value = self.policy.critic(*last_state).detach()
                values_ = torch.cat((values_, bootstrapped_value[0]), dim=0)
            advantages, returns = self.advantage_func(self.gamma, self._lambda, values_.detach(), masks, rewards)

        return advantages, returns

    def update(self, memory, batches):
        """
        This function implements the update step of the Proximal Policy Optimization (PPO) algorithm for a swarm of
        robots. It takes in the memory buffer containing the experiences of the swarm, as well as the number of batches
        to divide the experiences into for training. The function first computes the discounted rewards for each robot
        in the swarm and normalizes them. It then flattens the rewards and masks and converts them to PyTorch tensors.
        Next, the function retrieves the observations, actions, and log probabilities from the memory buffer and divides
        them into minibatches for training. For each minibatch, it calculates the advantages using the generalized
        advantage estimator (GAE) and trains the policy for K epochs using the surrogate loss function. The function
        then updates the weights of the actor and critic networks using the optimizer.
        Finally, the function copies the updated weights to the old policy for future use in the next update step.

        :param memory: The memory to update the network with.
        :param batch_size: The size of batches.
        """

        batch_size = len(memory)
        mini_batch_size = int(batch_size / batches)

        if batch_size > len(memory):
            warnings.warn("Batch size is larger than memory capacity. Setting batch size to memory capacity.")
            batch_size = len(memory)
        if batch_size == 1:
            raise ValueError("Batch size must be greater than 1.")

        # Unroll current and previous memory if there is any
        if len(memory.swarmMemory) > 0:
            p_states, p_actions, p_logprobs, p_rewards, p_masks = memory.unroll_memory(memory.swarmMemory)
            if p_rewards.numel() > 1:
                p_rewards = (p_rewards - p_rewards.mean()) / (p_rewards.std() + 1e-10)
            else:
                p_rewards = p_rewards / (torch.abs(p_rewards) + 1e-10)
        else:
            p_states, p_actions, p_logprobs, p_rewards, p_masks = [], [], [], [], []

        c_states, c_actions, c_logprobs, c_rewards, c_masks = memory.unroll_memory(memory.environmentMemory)

        if c_rewards.numel() > 1:
            c_rewards = (c_rewards - c_rewards.mean()) / (c_rewards.std() + 1e-10)
        else:
            c_rewards = c_rewards / (torch.abs(c_rewards) + 1e-10)

        # Calculate the bootstrapped advantages & returns for the last episode
        if c_rewards.numel() == 0:
            b_advantages, b_returns = torch.FloatTensor(np.array([])), torch.FloatTensor(np.array([]))
        else:
            b_advantages, b_returns = memory.calculate_bootstrapped_advantages_returns(self.get_advantages_returns)

        if len(p_states) == 0:
            states = c_states
            actions = c_actions
            old_logprobs = c_logprobs
            advantages = b_advantages
            returns = b_returns
        else:
            adv, ret = self.get_advantages_returns(p_states, p_actions, p_masks, p_rewards)

            states = tuple(torch.concat([p_states[i], c_states[i]]) for i in range(len(p_states)))
            actions = torch.concat([p_actions, c_actions])
            old_logprobs = torch.concat([p_logprobs, c_logprobs])

            advantages = torch.concat([adv, b_advantages])
            returns = torch.concat([ret, b_returns])

        # Logger
        log_values = []
        #TODO logger?
        #logger.add_reward([np.array(rewards.detach().cpu()).mean()])

        # Normalize rewards by running reward
        # self.running_reward_std.update(np.array(rewards))
        # rewards = np.clip(np.array(rewards) / self.running_reward_std.get_std(), -10, 10)
        # rewards = torch.tensor(rewards).type(torch.float32)

        # Save current weights if the mean reward is higher than the best reward so far
        # TODO FIX LOGGER
        # if logger.better_reward():
        #     print("Saving best weights with reward {}".format(logger.reward_best))
        #     torch.save(self.policy.state_dict(), 'best.pth')

        # Train policy for K epochs: sampling and updating
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, False):
                # Evaluate old actions and values using current policy
                batch_states = (states[0][index], states[1][index], states[2][index], states[3][index])
                batch_actions = actions[index]
                logprobs, values, dist_entropy = self.policy.evaluate(batch_states, batch_actions)
                log_values.append(values.detach().mean().item())
                # Importance ratio: p/q
                ratios = torch.exp(logprobs - old_logprobs[index].detach())

                # Actor loss using Surrogate loss
                surr1 = ratios * advantages[index]
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[index]
                entropy = 0.001 * dist_entropy
                actor_loss = ((-torch.min(surr1, surr2).type(torch.float32)) - entropy).mean()

                # TODO CLIP VALUE LOSS ? Probably not necessary as according to:
                # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
                critic_loss = self.MSE_loss(returns[index].squeeze(), values)
                # Total loss
                # loss = actor_loss + critic_loss
                # self.logger.add_loss(loss.detach().mean().item(), entropy=entropy.detach().mean().item(), critic_loss=critic_loss.detach().mean().item(), actor_loss=actor_loss.detach().mean().item())

                # Sanity checks
                if torch.isnan(actor_loss).any():
                    print(entropy.mean())
                    print(returns)
                    print(values)
                if torch.isnan(critic_loss).any():
                    print(entropy.mean())
                    print(returns)
                    print(values)
                assert not torch.isnan(actor_loss).any(), f"Actor loss is NaN: {actor_loss}"
                assert not torch.isinf(critic_loss).any()
                assert not torch.isinf(actor_loss).any()
                # Backward gradients
                self.optimizer_a.zero_grad()
                actor_loss.backward(retain_graph=True)
                # Global gradient norm clipping https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=0.5)
                self.optimizer_a.step()

                self.optimizer_c.zero_grad()
                critic_loss.backward()
                # Global gradient norm clipping https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html
                torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), max_norm=0.5)
                self.optimizer_c.step()
                # # Global gradient norm clipping https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html
                # torch.nn.utils.clip_grad_norm_(self.policy.ac.parameters(), max_norm=0.5)

        #logger.add_value([np.array(log_values).mean()])

        # Clear memory
        memory.clear_memory()