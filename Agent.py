import random
import tensorflow as tf
import keras as k


class Agent:
    def __init__(self):
        self.total_reward = 0.0

    def predict(self, obs, possible_actions):
        return random.choice(possible_actions), random.randint(0, 10)          # wählt zufällige Aktion
