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
        self.robotsCount = robotsCount
        self.swarmMemory = []
        self.environmentMemory = []
        self.currentTerminalStates = []
        self.init()
        self.relativeIndices = self.getRelativeIndices()

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

    def getObservationOfAllRobots(self):
        """
        This function returns the observations of all robots in the swarm.
        """
        laser = []
        orientation = []
        distance = []
        velocity = []
        for robotmemory in self.swarmMemory:
            for o in robotmemory.observations:
                laser.append(o[0])
                orientation.append(o[1])
                distance.append(o[2])
                velocity.append(o[3])

        return [torch.stack(laser), torch.stack(orientation), torch.stack(distance), torch.stack(velocity)]

    def getActionsOfAllRobots(self):
        actions = []
        for robotmemory in self.swarmMemory:
            for action in robotmemory.actions:
                actions.append(action)

        return actions

    def getLogProbsOfAllRobots(self):
        logprobs = []
        for robotmemory in self.swarmMemory:
            for logprob in robotmemory.logprobs:
                logprobs.append(logprob)

        return logprobs

    def clear_memory(self):
        for memory in self.swarmMemory:
            memory.clear_memory()
        self.swarmMemory = []

    def clear_episode(self):
        """
        This function clears the memory of the environment. It also resets the currentTerminalStates to False.
        """
        for memory in self.environmentMemory:
            memory.clear_memory()
        self.environmentMemory = self.environmentMemory[:self.robotsCount]
        self.currentTerminalStates = self.currentTerminalStates[:self.robotsCount]

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