from keras.models import Model
from keras.layers import Dense, Flatten, Input, Conv2D, Conv1D, ReLU
from keras.optimizers import Adam
import keras.backend as K
from collections import namedtuple

import random
import numpy as np
import math
from tqdm import tqdm

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))


class DQN:
    def __init__(self, act_dim, env_dim, args):

        self.current_step = 0

        # Replay Memory
        self.memory_capacity = args.memory_size
        self.memory = []
        self.push_count = 0

        # Epsilon Greedy Params
        self.start = args.eps_start
        self.end = args.eps_end
        self.decay = args.eps_decay

        self.policy_net = self._create_model(act_dim, env_dim, args.learningrate)
        self.target_net = self._create_model(act_dim, env_dim, args.learningrate)
        self.update_target_net(self.policy_net)

    def _create_model(self, act_dim, env_dim, lr):
        # [X Rob] [Y Rob] [lin Vel]
        # [X Goal] [Y Goal] [ang Vel]
        # [lin acc] [ang acc] [orientation]
        input_shape = Input(shape=env_dim)

        # conv = Conv1D(filters=6, kernel_size=2, strides=1, padding="valid")(input_shape)
        # conv = ReLU()(conv)
        # conv = Conv1D(filters=12, kernel_size=2, strides=1, padding="valid")(conv)
        # conv = ReLU()(conv)
        #
        # flatten = Flatten()(conv)
        flatten = Flatten()(input_shape)

        dense = Dense(units=64, kernel_initializer='random_normal', use_bias=False)(flatten)
        dense = ReLU()(dense)
        action = Dense(units=act_dim, kernel_initializer='random_normal', use_bias=False)(dense)
        model = Model(inputs=input_shape, outputs=action)

        adam = Adam(lr=lr)
        model.compile(optimizer=adam, loss='mse')
        return model

    def update_target_net(self, policy_net):
        weights = self.target_net.get_weights()
        policy_weights = policy_net.get_weights()
        for i in range(len(policy_weights)):
            weights[i] = policy_weights[i]
        self.target_net.set_weights(weights)

    ######### AGENT ########

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

    #############

    def save_weights(self, path):
        path += 'DQN'
        self.policy_net.save_weights(path + '_policy.h5')
        self.target_net.save_weights(path + '_target.h5')

    def train(self, env, args):

        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")
        for episode in tqdm_e:
            env.reset()
            state = env.get_observation()
            state = np.expand_dims(state, axis=0)
            # print(f'Episode: {episode}')
            # print(f'Steps Agent: {agent.current_step}')
            # print(f'Epsilon Greedy: {self.epsilon_greedy_strategy(self.current_step)}')
            while not env.is_done():
                possible_actions = env.get_actions()
                action = self.choose_action(state, possible_actions, self.policy_net)
                next_state, reward, _ = env.step(action)
                done = env.is_done()
                if K.is_tensor(next_state):
                    next_state = K.get_value(next_state)
                self.push_memory(
                    Experience(state=state, action=action, next_state=next_state, reward=reward, done=done))
                env.total_reward += reward
                state = next_state

                if self.can_provide_sample(args.batchsize):
                    experiences = self.sample_memory(args.batchsize)
                    states = []
                    targets = []
                    # print(len(experiences))
                    for sample in experiences:
                        _state, _action, _next_state, _reward, _done = sample
                        target = self.target_net.predict(_state)

                        if _done:
                            target[0][_action] = _reward
                        else:
                            Q_future = np.amax(self.target_net.predict(_next_state)[0])
                            target[0][_action] = _reward + Q_future * args.gamma
                        states.append(_state)
                        targets.append(target)
                    states = np.squeeze(np.asarray(states), axis=1)
                    targets = np.squeeze(np.asarray(targets), axis=1)
                    self.policy_net.fit(states, targets, batch_size=args.batchsize, epochs=1, verbose=0)

                # Display score
                tqdm_e.set_description("TODO!")
                tqdm_e.refresh()

            if episode % args.save_intervall == 0:
                self.save_weights(args.path)

            if episode % args.target_update == 0:
                self.update_target_net(self.policy_net)

            print("Total reward got: %.4f" % env.total_reward)
