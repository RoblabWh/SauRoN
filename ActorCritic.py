from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Input, Conv2D, Conv1D, ReLU, Add
from keras.optimizers import Adam
import random, math
import tensorflow as tf
from collections import deque

class ActorCritic:
    def __init__(self, env, lr, start, end, decay, sess, batch_size):

        self.env = env
        self.sess = sess
        self.start = start
        self.decay = decay
        self.end = end
        self.gamma = .95
        self.tau = .125
        self.batch_size = batch_size

        # Actor Model
        self.actor_state_input, self.actor_model = self._create_actor_model(lr)
        _, self.target_actor_model = self._create_actor_model(lr)

        self.actor_critic_grad = tf.placeholder(tf.float32,
                                  [None, self.env.get_actions.length]) # where we will feed de/dC from critic, env.action_space austauschen

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad) # dC/dA from actor
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(lr).apply_gradients(grads)

        # Critic Model
        self.critic_state_input, self.critic_action_input, self.critic_model = self._create_critic_model(lr)
        _, _, self.target_critic_model = self._create_critic_model(lr)

        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)  # where we calculate de/dC for feeding above
        self.sess.run(tf.initialize_all_variables())   # unsicher mit session

        self.current_step = 0

        # Memory
        self.memory = []

    def _create_actor_model(self, lr):
        # [X Rob] [Y Rob] [lin Vel]
        # [X Goal] [Y Goal] [ang Vel]
        # [lin acc] [ang acc] [orientation]
        state_input = Input(shape=(4, 7))

        flatten = Flatten()(state_input)

        dense = Dense(64, kernel_initializer='random_normal', use_bias=False)(flatten)
        dense = ReLU()(dense)
        action = Dense(units=3, kernel_initializer='random_normal', use_bias=False)(dense)

        model = Model(input=state_input, outputs=action)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def _create_critic_model(self, lr):
        state_input = Input(shape=(4, 7))
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=3)
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    # ------------------ #
    #   Model Training   #
    # ------------------ #

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([cur_state, action], reward, verbose=0)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, self.batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    # --------------------------- #
    #   Target Model Updating     #
    # --------------------------- #

    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    # ---------------------- #
    #   Model Predictions    #
    # ---------------------- #

    def epsilon_greedy_strategy(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)

    def act(self, cur_state, possible_actions):
        rate = self.epsilon_greedy_strategy(self.current_step)
        self.current_step += 1

        # Exploration Exploitation Trade Off
        if rate > random.random():
            # print("CHOSE EXPLORATION")
            return random.choice(possible_actions)  # explore

        else:
            # print("CHOSE EXPLOITATION")
            return self.actor_model.predict(cur_state)