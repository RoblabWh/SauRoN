import sys
from old import DQN
from deprecated import Agent, Environment
import numpy as np
from PyQt5.QtWidgets import QApplication
from collections import namedtuple
import keras.backend as K
import tensorflow as tf


# HYPERPARAMETERS
batch_size = 40
gamma = 0.8
eps_start = 1
eps_end = 0.01
eps_decay = 0.0001
target_update = 10
memory_size = 10000
lr = 0.001
num_episodes = 1000
steps_left = 200

# Workaround for not getting error message
#def except_hook(cls, exception, traceback):
#    sys.__excepthook__(cls, exception, traceback)

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))


def main():
    agent = Agent.Agent(eps_start, eps_end, eps_decay, memory_size)

    # TODO: Netze noch implementieren, target_net ist Kopie vom policy_net
    policy_net = DQN.DQN(lr)
    target_net = DQN.DQN(lr)
    target_net.update_target_net(policy_net)

    # print(tf.__version__)   # Test für Tensorflow

    app = QApplication(sys.argv)
    env = Environment.Environment(app, steps_left)

    for episode in range(num_episodes):
        env.reset()
        state = env.get_observation()
        state = np.expand_dims(state, axis=0)
        print(f'Episode: {episode}')
        # print(f'Steps Agent: {agent.current_step}')
        print(f'Epsilon Greedy: {agent.epsilon_greedy_strategy(agent.current_step)}')
        while not env.is_done():
            possible_actions = env.get_actions()
            action = agent.choose_action(state, possible_actions, policy_net)
            next_state, reward = env.step(action)
            done = env.is_done()
            if K.is_tensor(next_state):
                next_state = K.get_value(next_state)
            agent.push_memory(Experience(state=state, action=action, next_state=next_state, reward=reward, done=done))
            env.total_reward += reward
            state = next_state

            if agent.can_provide_sample(batch_size):
                experiences = agent.sample_memory(batch_size)
                states = []
                targets = []
                # print(len(experiences))
                for sample in experiences:
                    _state, _action, _next_state, _reward, _done = sample

                    if _done:
                        target[0][_action] = _reward
                    else:
                        target = target_net.model.predict(_next_state)
                        Q_future = np.amax(target_net.model.predict(_next_state)[0])
                        target[0][_action] = _reward + Q_future * gamma
                    states.append(_state)
                    targets.append(target)
                states = np.squeeze(np.asarray(states), axis=1)
                targets = np.squeeze(np.asarray(targets), axis=1)
                policy_net.model.fit(states, targets, batch_size=batch_size, epochs=1, verbose=0)

        if episode % target_update == 0:
            target_net.update_target_net(policy_net)

        print("Total reward got: %.4f" % env.total_reward)
    # sys.exit(app.exec_())
    # sys.excepthook = except_hook

def main2():
    # unsicher mit Session
    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    # K.set_session(sess)

    app = QApplication(sys.argv)
    env = Environment.Environment(app, steps_left)

    actor_critic = ActorCritic.ActorCritic(env, lr, eps_start, eps_end, eps_decay, sess, batch_size)

    actor_critic.critic_model.summary()
    #
    # return

    for episode in range(num_episodes):
        env.reset()
        cur_state = env.get_observation()
        cur_state = np.expand_dims(cur_state, axis=0)
        print(f'Episode: {episode}')
        # print(f'Steps Agent: {agent.current_step}')
        # print(f'Epsilon Greedy: {agent.epsilon_greedy_strategy(agent.current_step)}')
        while not env.is_done():
            possible_actions = env.get_actions()
            action = actor_critic.act(cur_state, possible_actions)
            action = np.asarray([action, action, action])
            action = action.reshape((1, 3))
            next_state, reward = env.step(action)
            done = env.is_done()
            if K.is_tensor(next_state):
                next_state = K.get_value(next_state)
            actor_critic.remember(cur_state, action, reward, next_state, done)
            env.total_reward += reward
            actor_critic.train()
            cur_state = next_state

        if episode % target_update == 0:
            actor_critic.update_target()

        print("Total reward got: %.4f" % env.total_reward)
    # sys.exit(app.exec_())
    # sys.excepthook = except_hook

if __name__ == '__main__':
    main()