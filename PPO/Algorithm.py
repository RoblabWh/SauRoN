import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch
import os, warnings
import numpy as np
from pathlib import Path
from utils import initialize_output_weights, RunningMeanStd
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from PPO.BigInput import BigInput
from PPO.SmallInput import SmallInput

from utils import statesToObservationsTensor, normalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    """
    A PyTorch Module that represents the actor network of a PPO agent.

    This module takes in four inputs: a lidar scan, orientation to goal, distance to goal, and velocity.
    It then applies convolutional and dense layers to each input separately and concatenates the outputs
    to produce a flattened feature vector that can be fed into a downstream neural network.

    :param scan_size: The number of lidar scans in the input lidar scan.
    """
    def __init__(self, scan_size, inputspace):
        super(Actor, self).__init__()
        if inputspace == 'big':
            self.Inputspace = BigInput(scan_size)
        elif inputspace == 'small':
            self.Inputspace = SmallInput(scan_size)

        # Mu
        self.mu = nn.Linear(in_features=self.Inputspace.out_features, out_features=2)
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
    def __init__(self, scan_size, inputspace):
        super(Critic, self).__init__()
        if inputspace == 'big':
            self.Inputspace = BigInput(scan_size)
        elif inputspace == 'small':
            self.Inputspace = SmallInput(scan_size)

        # Value
        self.value = nn.Linear(in_features=self.Inputspace.out_features, out_features=1)
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
    def __init__(self, scan_size, inputspace, logger):
        super(ActorCritic, self).__init__()
        action_dim = 2
        self.actor_cnt = 0
        self.critic_cnt = 0
        self.logger = logger
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(scan_size, inputspace)
        self.critic = Critic(scan_size, inputspace)

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

    def __init__(self, scan_size, inputspace, lr, betas, gamma, _lambda, K_epochs, eps_clip, logger, restore=False, ckpt=None, advantages_func=None):
        # Algorithm parameters
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self._lambda = _lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Folder the models are stored in
        if not restore:
            if os.path.isfile(ckpt):
                raise ValueError("Model path must be a directory, not a file")
            if not os.path.exists(ckpt):
                print(f"Creating model directory under {os.path.abspath(ckpt)}")
                os.makedirs(ckpt)
        self.model_path = Path(ckpt)

        # Current Policy
        self.policy = ActorCritic(scan_size, inputspace, logger).to(device)
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

    def get_advantages(self, values, masks, rewards):
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
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self._lambda * masks[i] * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        advantages = torch.FloatTensor(advantages).to(device)
        norm_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-10) if advantages.numel() > 1 else advantages

        returns = torch.FloatTensor(returns).to(device)

        assert not torch.isnan(norm_adv).any(), f"Advantages are NaN: {norm_adv}"

        return norm_adv, returns

    # def get_advantages_returns(self, states, actions, masks, rewards):
    #     # Advantages
    #     with torch.no_grad():
    #         _, values_, _ = self.policy.evaluate(states, actions)
    #         if values_.dim() == 0:
    #             values_ = values_.unsqueeze(0)
    #         if masks[-1] == 0:
    #             last_state = (states[0][-1].unsqueeze(0),
    #                           states[1][-1].unsqueeze(0),
    #                           states[2][-1].unsqueeze(0),
    #                           states[3][-1].unsqueeze(0))
    #             bootstrapped_value = self.policy.critic(*last_state).detach()
    #             values_ = torch.cat((values_, bootstrapped_value[0]), dim=0)
    #         advantages, returns = self.advantage_func(self.gamma, self._lambda, values_.detach(), masks, rewards)
    #
    #     return advantages, returns

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

        memory.unroll_last_episode(0)

        batch_size = len(memory)
        mini_batch_size = int(batch_size / batches)

        if batch_size > len(memory):
            warnings.warn("Batch size is larger than memory capacity. Setting batch size to memory capacity.")
            batch_size = len(memory)
        if batch_size == 1:
            raise ValueError("Batch size must be greater than 1.")

        states, next_states, actions, old_logprobs, rewards, dones = memory.to_tensor()

        self.running_reward_std.update(torch.cat(rewards).detach().cpu().numpy())
        rewards = [reward / self.running_reward_std.get_std() for reward in rewards]

        # Advantages
        with torch.no_grad():
            advantages = []
            returns = []
            for i in range(len(states)):
                _, values_, _ = self.policy.evaluate(states[i], actions[i])
                if dones[i][-1] == 1:
                    laser, orientation, distance, velocity = tuple(obs[-1].unsqueeze(0) for obs in next_states[i])
                    bootstrapped_value = self.policy.critic(laser.to(self.device), orientation.to(self.device), distance.to(self.device), velocity.to(self.device)).detach()
                    # TODO hier nochmal guckne next_obs ist wahrscheinlich quatsch
                    if values_.dim() != bootstrapped_value.squeeze(0).dim():
                        values_ = values_.unsqueeze(0)
                    if values_.dim() == 0:
                        print("wtf")
                    values_ = torch.cat((values_, bootstrapped_value.squeeze(0)), dim=0)
                else:
                    if values_.size() == torch.Size([]):
                        values_ = values_.unsqueeze(0)
                    values_ = torch.cat((values_, torch.tensor([0.0]).to(device)), dim=0)
                adv, ret = self.get_advantages(values_.detach(), dones[i], rewards[i].detach())
                advantages.append(adv)
                returns.append(ret)

        # Merge all agent states, actions, rewards etc.
        advantages = torch.cat(advantages)
        returns = torch.cat(returns)
        actions = torch.cat(actions)
        old_logprobs = torch.cat(old_logprobs)
        states_ = tuple()
        for i in range(len(states[0])):
            states_ += (torch.cat([states[k][i] for k in range(len(states))]),)
        states = states_

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
                assert not torch.isnan(critic_loss).any(), f"Critic loss is NaN: {critic_loss}"
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