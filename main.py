import Environment, Agent, sys, Network
import tensorflow as tf
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
target_update = 3
memory_size = 10000
lr = 0.001
num_episodes = 1000
steps_left = 200

# Workaround for not getting error message
#def except_hook(cls, exception, traceback):
#    sys.__excepthook__(cls, exception, traceback)

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))


def main():
    agent = Agent.Agent(eps_start, eps_end, eps_decay, memory_size)

    # TODO: Netze noch implementieren, target_net ist Kopie vom policy_net
    policy_net = Network.DQN(lr)
    target_net = Network.DQN(lr)
    target_net.update_target_net(policy_net)

    # print(tf.__version__)   # Test für Tensorflow

    app = QApplication(sys.argv)
    env = Environment.Environment(app, steps_left)

    for episode in range(num_episodes):
        env.reset()
        state = env.get_observation()
        state = np.expand_dims(state, axis=0)
        print(f'Episode: {episode}')
        while not env.is_done():
            possible_actions = env.get_actions()
            action = agent.choose_action(state, possible_actions, policy_net)
            next_state, reward = env.step(action)
            if K.is_tensor(next_state):
                next_state = K.get_value(next_state)
            agent.push_memory(Experience(state=state, action=action, next_state=next_state, reward=reward))
            env.total_reward += reward
            state = next_state

            if agent.can_provide_sample(batch_size):
                experiences = agent.sample_memory(batch_size)
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


if __name__ == '__main__':
    main()