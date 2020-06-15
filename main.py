import Environment, Agent, sys, ReplayMemory, Network
import tensorflow as tf
import EpsilonGreedyStrategy
import numpy as np
from PyQt5.QtWidgets import QApplication
from collections import namedtuple
import keras.backend as K

# HYPERPARAMETERS
batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 1000

# Workaround for not getting error message
#def except_hook(cls, exception, traceback):
#    sys.__excepthook__(cls, exception, traceback)


Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))


def main():
    strategy = EpsilonGreedyStrategy.EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = Agent.Agent(strategy)
    memory = ReplayMemory.ReplayMemory(memory_size)

    # TODO: Netze noch implementieren, target_net ist Kopie vom policy_net
    policy_net = Network.DQN(lr)
    target_net = Network.DQN(lr)
    episode_durations = []

    print(tf.__version__)   # Test für Tensorflow

    app = QApplication(sys.argv)
    env = Environment.Environment(app)

    for episode in range(num_episodes):
        env.reset()
        state = env.get_observation()
        state = np.expand_dims(state, axis=0)
        #state = np.zeros(shape=(1, 4, 3, 3))

        while not env.is_done():
            # obs = env.get_observation()
            possible_actions = env.get_actions()
            action = agent.choose_action(state, possible_actions, policy_net)
            print("Gewaehlte Aktion: " + str(action))
            next_state, reward, done = env.step(action)
            if K.is_tensor(next_state):
                next_state = K.get_value(next_state)
            memory.push(Experience(state=state, action=action, next_state=next_state, reward=reward))
            agent.total_reward += reward
            state = next_state

            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences)

                # evtl. besser in Netzwerk Klasse?
                # current_q_values = QValues.QValues.get_current(policy_net, states, actions)
                # next_q_values = QValues.QValues.get_next(target_net, next_states)
                # target_q_values = (next_q_values * gamma) + rewards

                # PyTorch spezifisch -> evtl besser in Netzwerkklasse?
                # loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

        if episode % target_update == 0:
            # TODO: Uebertrage die neuen Weights des Policy Networks auf das target_net
            pass

        print("Total reward got: %.4f" % agent.total_reward)
    # sys.exit(app.exec_())
    # sys.excepthook = except_hook


def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))

    t1 = K.concatenate(batch.state, axis=0)
    t2 = K.concatenate(batch.action, axis=0)
    t3 = K.concatenate(batch.reward, axis=0)
    t4 = K.concatenate(batch.next_state, axis=0)

    return t1, t2, t3, t4


if __name__ == '__main__':
    main()