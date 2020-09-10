import random
import numpy as np
import math
import tensorflow as tf
import keras as k

# Datei kann weg -> alles im ActorCritic drin

class AgentActorCritic:
    def __init__(self, start, end, decay, capacity):
        self.current_step = 0

        #Replay Memory
        self.memory_capacity = capacity
        self.memory = []
        self.push_count = 0

        #Epsilon Greedy Params
        self.start = start
        self.end = end
        self.decay = decay

    def push_memory(self, experience):
        if len(self.memory) < self.memory_capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.memory_capacity] = experience
        self.push_count += 1

    def sample_memory(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

    def epsilon_greedy_strategy(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)

    def choose_action(self, state, possible_actions, policy_net):
        rate = self.epsilon_greedy_strategy(self.current_step)
        self.current_step += 1

        # Exploration Exploitation Trade Off
        if rate > random.random():
            # print("CHOSE EXPLORATION")
            return random.choice(possible_actions)  # explore

        else:
            # print("CHOSE EXPLOITATION")
            test = policy_net.predict(state)[0]
            return np.argmax(test)