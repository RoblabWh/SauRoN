import torch
import numpy as np
import copy
from PPO.Memory import Memory

class SwarmMemory:
    """
    This class is used to store the memory of the environment. It is used to store the observations, actions, rewards,
    logprobs, reached_goal and is_terminal. It is used to store the memory of the environment for each robot.

    :param robotsCount: The number of robots in the environment
    """
    def __init__(self, robotsCount = 0):
        self.ready_to_train = False
        self.robotsCount = robotsCount
        self.swarmMemory = []
        self.environmentMemory = []
        self.currentTerminalStates = []
        self.init()
        self.relativeIndices = self.getRelativeIndices()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init(self):
        self.environmentMemory = [Memory() for _ in range(self.robotsCount)]
        self.currentTerminalStates = [False for _ in range(self.robotsCount)]

    def __getitem__(self, item):
        return self.environmentMemory[item]

    # Gets relative Index according to currentTerminalStates
    def getRelativeIndices(self):
        """
        This function returns the indices of the robots that are not in a terminal state.
        """
        relativeIndices = []
        for i in range(len(self.currentTerminalStates)):
            if not self.currentTerminalStates[i]:
                relativeIndices.append(i)

        return relativeIndices

    def insertObservations(self, laser, orientation, distance, velocity):
        for i in range(len(self.relativeIndices)):
            self.environmentMemory[self.relativeIndices[i]].observations.append([laser[i], orientation[i], distance[i], velocity[i]])

    def insertAction(self, action):
        for i in range(len(self.relativeIndices)):
            self.environmentMemory[self.relativeIndices[i]].actions.append(action[i])

    def insertReward(self, reward):
        for i in range(len(self.relativeIndices)):
            self.environmentMemory[self.relativeIndices[i]].rewards.append(reward[i])

    def insertLogProb(self, logprob):
        for i in range(len(self.relativeIndices)):
            self.environmentMemory[self.relativeIndices[i]].logprobs.append(logprob[i])

    def insertReachedGoal(self, reachedGoal, isTerminal):
        terminalGoal = np.logical_and(reachedGoal, isTerminal)
        for idx in np.where(isTerminal)[0]:
            self.environmentMemory[self.relativeIndices[idx]].reached_goal.append(terminalGoal[idx])

    def insertIsTerminal(self, isTerminal):
        """
        This function inserts the isTerminal values in the memory. It also checks if all robots are in a terminal state.
        If so, it resets the currentTerminalStates to False.

        :param isTerminal: The isTerminal values of the robots
        """
        for i in range(len(self.relativeIndices)):
            self.environmentMemory[self.relativeIndices[i]].is_terminals.append(isTerminal[i])
            if isTerminal[i]:
                self.currentTerminalStates[self.relativeIndices[i]] = True
        self.relativeIndices = self.getRelativeIndices()

        # check if currentTerminalStates is all True
        if all(self.currentTerminalStates):
            self.currentTerminalStates = [False for _ in range(len(self.currentTerminalStates))]

    def copyMemory(self):
        for memory in self.environmentMemory:
            self.swarmMemory.append(copy.deepcopy(memory))

    def getObservations(self, memory):
        """
        This function returns the observations of all robots in the swarm.
        """
        laser = []
        orientation = []
        distance = []
        velocity = []

        for robotmemory in memory:
            for o in robotmemory.observations:
                laser.append(o[0])
                orientation.append(o[1])
                distance.append(o[2])
                velocity.append(o[3])
        return [torch.stack(laser), torch.stack(orientation), torch.stack(distance), torch.stack(velocity)]

    def getActions(self, memory):
        actions = []
        for robotmemory in memory:
            for action in robotmemory.actions:
                actions.append(action)
        return actions

    def getLogProbs(self, memory):
        logprobs = []
        for robotmemory in memory:
            for logprob in robotmemory.logprobs:
                logprobs.append(logprob)
        return logprobs

    def getRewards(self, memory):
        rewards = []
        for robotmemory in memory:
            for reward in robotmemory.rewards:
                rewards.append(reward)
        return rewards

    def getTerminalStates(self, memory):
        terminalStates = []
        for robotmemory in memory:
            for terminalState in robotmemory.is_terminals:
                terminalStates.append(terminalState)
        return terminalStates

    def to_tensor(self, states, actions, logprobs, rewards, masks):

        return tuple(torch.FloatTensor(state).to(self.device) for state in states), \
            torch.stack(actions).to(self.device), \
            torch.FloatTensor(logprobs).to(self.device), \
            torch.FloatTensor(rewards).to(self.device), \
            torch.FloatTensor(masks).to(self.device)

    def calculate_bootstrapped_advantages_returns(self, func):
        adv = []
        returns = []
        for mem in self.environmentMemory:
            states, actions, _, rewards, masks = self.unroll_memory(mem)
            if rewards.numel() > 1:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)
            elif rewards.numel() == 1:
                rewards = rewards / (torch.abs(rewards) + 1e-10)
            else:
                continue
            a, r = func(states, actions, masks, rewards)
            adv.append(a)
            returns.append(r)
        return torch.concat(adv), torch.concat(returns)

    def clear_memory(self):
        for memory in self.swarmMemory:
            memory.clear_memory()
        self.swarmMemory = []
        self.ready_to_train = False

    def clear_episode(self):
        """
        This function clears the memory of the environment. It also resets the currentTerminalStates to False.
        """
        for memory in self.environmentMemory:
            memory.clear_memory()
        self.environmentMemory = self.environmentMemory[:self.robotsCount]
        self.currentTerminalStates = self.currentTerminalStates[:self.robotsCount]

    def unroll_memory(self, memory):
        if not isinstance(memory, list):
            memory = [memory]
        non_empty_memory = [mem for mem in memory if len(mem) > 0]
        if len(non_empty_memory) == 0:
            return torch.FloatTensor(np.array([])), torch.FloatTensor(np.array([])), \
                   torch.FloatTensor(np.array([])), torch.FloatTensor(np.array([])), \
                   torch.FloatTensor(np.array([]))
        states = self.getObservations(non_empty_memory)
        actions = self.getActions(non_empty_memory)
        logprobs = self.getLogProbs(non_empty_memory)
        rewards = self.getRewards(non_empty_memory)
        masks = self.getTerminalStates(non_empty_memory)

        return self.to_tensor(states, actions, logprobs, rewards, masks)

    def __add__(self, other):
        new_memory = SwarmMemory()
        new_memory.environmentMemory += self.environmentMemory
        new_memory.currentTerminalStates += self.currentTerminalStates
        if other is not None:
            new_memory.environmentMemory += other.environmentMemory
            new_memory.currentTerminalStates += other.currentTerminalStates
        return new_memory

    def __iadd__(self, other):
        if other is not None:
            self.environmentMemory += other.environmentMemory
            self.currentTerminalStates += other.currentTerminalStates
        return self

    def __len__(self):
        length = 0
        for memory in self.environmentMemory:
            length += len(memory)
        for memory in self.swarmMemory:
            length += len(memory)
        return length