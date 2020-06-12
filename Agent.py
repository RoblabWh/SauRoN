import random
import numpy as np
import tensorflow as tf
import keras as k


class Agent:
    def __init__(self, strategy):
        self.total_reward = 0.0
        self.current_step = 0
        self.strategy = strategy

    def predict(self, state, possible_actions, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        print(rate)
        # Exploration Exploitation Trade Off
        if rate > random.random():
            return random.choice(possible_actions)   # explore
        else:
            return np.argmax(policy_net(state)[0])     # exploit


        # return random.choice(possible_actions), random.randint(0, 10)          # wählt zufällige Aktion
