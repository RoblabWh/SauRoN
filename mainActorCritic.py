import Environment, Agent, sys, Network, ActorCritic
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
from PyQt5.QtWidgets import QApplication
from collections import namedtuple
import keras.backend as K

tf.disable_v2_behavior()

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

# Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))


def main():
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

        # if episode % target_update == 0:
        #     actor_critic.update_target()

        print("Total reward got: %.4f" % env.total_reward)
    # sys.exit(app.exec_())
    # sys.excepthook = except_hook


if __name__ == '__main__':
    main()