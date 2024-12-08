import torch
import numpy as np
import copy

class SwarmMemory(object):
    def __init__(self, num_agents=2, action_dim=2, max_size=int(1e5)):
        self.action_dim = action_dim
        self.max_size = max_size
        self.num_agents = num_agents
        self.memory = [Memory(action_dim=action_dim, max_size=max_size) for _ in range(num_agents)]
        self.past_memories = []

    def unroll_last_episode(self, num_robots):
        if len(self) > 0:
            self.past_memories.append(copy.deepcopy(self.memory))
            for i in range(self.num_agents):
                self.memory[i].clear_memory()
        self.num_agents = num_robots
        self.memory = [Memory(action_dim=self.action_dim, max_size=self.max_size) for _ in range(self.num_agents)]

    def get_agent_state(self, state, agent_id):
        tuple_state = tuple()
        for state_ in state:
            if isinstance(state_[agent_id], np.ndarray):
                tuple_state += (np.expand_dims(state_[agent_id], axis=0),)
            elif isinstance(state_[agent_id], torch.Tensor):
                tuple_state += (state_[agent_id].unsqueeze(0),)
            else:
                import warnings
                warnings.warn("State type not recognized")
        return tuple_state

    def add(self, state, action, action_logprobs, reward, done):
        for i in range(self.num_agents):
            self.memory[i].add(self.get_agent_state(state, agent_id=i), action[i], action_logprobs[i], reward[i], done[i])

    def __len__(self):
        length = 0
        for memories in self.past_memories:
            for memory in memories:
                length += len(memory)
        for i in range(self.num_agents):
            length += len(self.memory[i])
        return length

    def to_tensor(self):
        states, actions, logprobs, rewards, not_dones = [], [], [], [], []
        for memories in self.past_memories:
            for memory in memories:
                state, action, logprob, reward, not_done = memory.to_tensor()
                states.append(state)
                actions.append(action)
                logprobs.append(logprob)
                rewards.append(reward)
                not_dones.append(not_done)
        return states, actions, logprobs, rewards, not_dones

    def change_horizon(self, new_horizon):
        for i in range(self.num_agents):
            self.memory[i].change_horizon(new_horizon)

    def clear_memory(self):
        self.past_memories = []
        for i in range(self.num_agents):
            self.memory[i].clear_memory()


class Memory(object):
    def __init__(self, action_dim=3, max_size=int(1e5)):
        self.max_size = max_size
        self.action_dim = action_dim
        self.ptr = 0
        self.batch_ptr = 0
        self.size = 0

        self.state = [0 for _ in range(max_size)]
        self.action = np.zeros((max_size, action_dim))
        self.logprobs = np.zeros((max_size,))
        self.reward = np.zeros((max_size,))
        self.not_done = np.zeros((max_size,))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return self.size

    def add(self, state, action, action_logprobs, reward, done):
        self.state[self.ptr] = tuple(s for s in state)
        self.action[self.ptr] = action
        self.logprobs[self.ptr] = action_logprobs
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def to_tensor(self):
        return tuple(torch.FloatTensor(np.array(state)).squeeze(1).to(self.device) for state in zip(*self.state[:self.size])), \
               torch.FloatTensor(self.action[:self.size]).to(self.device), \
               torch.FloatTensor(self.logprobs[:self.size]).to(self.device), \
               torch.FloatTensor(self.reward[:self.size]).to(self.device), \
               torch.FloatTensor(self.not_done[:self.size]).to(self.device)

    def change_horizon(self, new_horizon):
        self.max_size = new_horizon
        self.state = [0 for _ in range(self.max_size)]
        self.action = np.zeros((self.max_size, self.action_dim))
        self.logprobs = np.zeros((self.max_size,))
        self.reward = np.zeros((self.max_size,))
        self.not_done = np.zeros((self.max_size,))
        self.clear_memory()

    def clear_memory(self):
        self.ptr = 0
        self.batch_ptr = 0
        self.size = 0

        self.state = [0 for _ in range(self.max_size)]
        self.action.fill(0)
        self.logprobs.fill(0)
        self.reward.fill(0)
        self.not_done.fill(0)