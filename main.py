import Environment, Agent, sys, ReplayMemory, Network
import tensorflow as tf
import EpsilonGreedyStrategy
import numpy as np
from PyQt5.QtWidgets import QApplication
from collections import namedtuple
import keras.backend as K

# HYPERPARAMETERS
batch_size = 10
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
    target_net.update_target_net(policy_net)

    # print(tf.__version__)   # Test für Tensorflow

    app = QApplication(sys.argv)
    env = Environment.Environment(app)

    for episode in range(num_episodes):
        env.reset()
        state = env.get_observation()
        state = np.expand_dims(state, axis=0)
        print(f'Episode: {episode}')
        print(env.done)
        while not env.is_done():
            # obs = env.get_observation()
            possible_actions = env.get_actions()
            action = agent.choose_action(state, possible_actions, policy_net)
            # print("Gewaehlte Aktion: " + str(action))
            next_state, reward = env.step(action)
            if K.is_tensor(next_state):
                next_state = K.get_value(next_state)
            memory.push(Experience(state=state, action=action, next_state=next_state, reward=reward))
            env.total_reward += reward
            state = next_state

            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                # print(len(experiences))
                for sample in experiences:
                    _state, _action, _next_state, _reward = sample

                    target = target_net.model.predict(_next_state)
                    Q_future = np.amax(target_net.model.predict(_next_state)[0])
                    target[0][_action] = _reward + Q_future * gamma

                    policy_net.model.fit(_state, target, epochs=1, verbose=0)

        if episode % target_update == 0:
            target_net.update_target_net(policy_net)

        print("Total reward got: %.4f" % env.total_reward)
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