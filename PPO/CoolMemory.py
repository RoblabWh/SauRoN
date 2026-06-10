import torch
import numpy as np

class SwarmMemory(object):
    def __init__(self, num_agents=2, action_dim=2, max_size=int(1e5)):
        self.action_dim = action_dim
        self.max_size = max_size
        self.num_agents = num_agents
        self.memory = [Memory(action_dim=action_dim, max_size=max_size) for _ in range(num_agents)]
        self.past_memories = []

    def unroll_last_episode(self, num_robots):
        # Hand the current per-agent memories over to the archive (no deepcopy --
        # fresh Memory objects are created below, so the references can move).
        non_empty = [memory for memory in self.memory if len(memory) > 0]
        if non_empty:
            for memory in non_empty:
                memory.trim()
            self.past_memories.append(non_empty)
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

    def add(self, state, next_state, action, action_logprobs, reward, done, agent_ids=None):
        """
        Adds one transition per (still active) agent. The state/action/... lists are
        compacted to the active agents; agent_ids maps each position to the original
        agent slot so robots can drop out mid-episode.
        """
        if agent_ids is None:
            agent_ids = range(self.num_agents)
        for i, agent in enumerate(agent_ids):
            self.memory[agent].add(self.get_agent_state(state, agent_id=i), self.get_agent_state(next_state, agent_id=i), action[i], action_logprobs[i], reward[i], done[i])

    def __len__(self):
        length = 0
        for memories in self.past_memories:
            for memory in memories:
                length += len(memory)
        for i in range(self.num_agents):
            length += len(self.memory[i])
        return length

    def to_tensor(self):
        states, next_obs, actions, logprobs, rewards, not_dones = [], [], [], [], [], []
        for memories in self.past_memories:
            for memory in memories:
                if len(memory) == 0:
                    continue
                state, last_next_state, action, logprob, reward, not_done = memory.to_tensor()
                states.append(state)
                next_obs.append(last_next_state)
                actions.append(action)
                logprobs.append(logprob)
                rewards.append(reward)
                not_dones.append(not_done)
        return states, next_obs, actions, logprobs, rewards, not_dones

    def change_horizon(self, new_horizon):
        for i in range(self.num_agents):
            self.memory[i].change_horizon(new_horizon)

    def clear_memory(self):
        self.past_memories = []
        self.memory = [Memory(action_dim=self.action_dim, max_size=self.max_size) for _ in range(self.num_agents)]


class Memory(object):
    def __init__(self, action_dim=3, max_size=int(1e5)):
        self.max_size = max_size
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0

        self.state = [0 for _ in range(max_size)]
        # only the latest next-observation is kept -- it is exclusively used to
        # bootstrap the value of a truncated episode segment
        self.last_next_state = None
        self.action = np.zeros((max_size, action_dim))
        self.logprobs = np.zeros((max_size,))
        self.reward = np.zeros((max_size,))
        self.not_done = np.zeros((max_size,))

    def __len__(self):
        return self.size

    def add(self, state, next_state, action, action_logprobs, reward, done):
        self.state[self.ptr] = tuple(s for s in state)
        self.last_next_state = tuple(s for s in next_state)
        self.action[self.ptr] = action
        self.logprobs[self.ptr] = action_logprobs
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def trim(self):
        """Slices the pre-allocated buffers down to the actually used size, so an
        archived memory does not hold on to its empty pre-allocation."""
        self.state = self.state[:self.size]
        self.action = self.action[:self.size]
        self.logprobs = self.logprobs[:self.size]
        self.reward = self.reward[:self.size]
        self.not_done = self.not_done[:self.size]
        self.max_size = max(self.size, 1)

    def to_tensor(self):
        # CPU tensors by design: the full rollout never moves to the GPU at once,
        # the update loop transfers minibatches on demand.
        return tuple(torch.FloatTensor(np.array(state)).squeeze(1) for state in zip(*self.state[:self.size])), \
               self.last_next_state, \
               torch.FloatTensor(self.action[:self.size]), \
               torch.FloatTensor(self.logprobs[:self.size]), \
               torch.FloatTensor(self.reward[:self.size]), \
               torch.FloatTensor(self.not_done[:self.size])

    def change_horizon(self, new_horizon):
        self.max_size = new_horizon
        self.clear_memory()

    def clear_memory(self):
        self.ptr = 0
        self.size = 0

        self.state = [0 for _ in range(self.max_size)]
        self.last_next_state = None
        self.action = np.zeros((self.max_size, self.action_dim))
        self.logprobs = np.zeros((self.max_size,))
        self.reward = np.zeros((self.max_size,))
        self.not_done = np.zeros((self.max_size,))
