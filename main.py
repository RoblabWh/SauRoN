import Environment, Agent, sys, ReplayMemory
import tensorflow as tf
import EpsilonGreedyStrategy
import numpy as np
from PyQt5.QtWidgets import QApplication
from collections import namedtuple

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
    policy_net = 0
    target_net = 0
    episode_durations = []

    print(tf.__version__)   # Test für Tensorflow

    app = QApplication(sys.argv)
    env = Environment.Environment(app)

    for episode in range(num_episodes):
        env.reset()
        state = env.get_observation()

        while not env.is_done():
            # obs = env.get_observation()
            possible_actions = env.get_actions()
            action = agent.predict(state, possible_actions, policy_net)
            print("Gewaehlte Aktion: " + str(action))
            next_state, reward, done = env.step(action)
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

    states = np.extract(batch.state, batch)
    actions = np.extract(batch.action, batch)
    rewards = np.extract(batch.reward, batch)
    next_states = np.extract(batch.next_state, batch)

    return states, actions, rewards, next_states


if __name__ == '__main__':
    main()