import numpy as np

from tqdm import tqdm
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import RMSprop, Adam

from utils import AverageMeter


class A2C:
    """ Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, args):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.gamma = args.gamma
        self.lr = args.learningrate
        # Create actor and critic networks
        self.shared = self.buildNetwork()
        self.actor = self.buildActor(self.shared)
        self.critic = self.buildCritic(self.shared)

        # Compile Models
        self.actor.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr))
        self.critic.compile(loss='mse', optimizer=Adam(lr=self.lr))
        # self.actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=self.lr))
        # self.critic.compile(loss='mse', optimizer=RMSprop(lr=self.lr))
        self.av_meter = AverageMeter()

        # self.a_opt = self.actor.optimizer()
        # self.c_opt = self.critic.optimizer()

    def buildNetwork(self):
        """ Assemble shared layers
        """
        inp = Input(self.env_dim)
        x = Flatten()(inp)
        x = Dense(64, activation='relu')(x) #64
        x = Dense(128, activation='relu')(x) #128
        return Model(inp, x)

    def buildActor(self, network):
        x = Dense(128, activation='relu')(network.output) #128
        out = Dense(self.act_dim, activation='softmax')(x)
        return Model(network.input, out)

    def buildCritic(self, network):
        x = Dense(128, activation='relu')(network.output) #128
        out = Dense(1, activation='linear')(x)
        return Model(network.input, out)

    def policy_action(self, s):
        """ Use the actor to predict the next action to take, using the policy
        """
        return np.random.choice(np.arange(self.act_dim), 1, p=self.actor.predict(s).ravel())[0]

    def discount(self, r):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r = np.zeros_like(r, dtype=float)
        cumul_r = 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def train_models(self, states, actions, rewards, done):
        """ Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(rewards)
        states = np.vstack(states)
        state_values = self.critic.predict(np.asarray(states))[:,0]
        advantages = discounted_rewards - np.reshape(state_values, len(state_values))
        # Networks optimization
        # self.a_opt([states, actions, advantages])
        # self.c_opt([states, discounted_rewards])
        actions = np.vstack(actions)
        self.actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        self.critic.fit(states, discounted_rewards, epochs=1, verbose=0)

    def train(self, env, args):
        """ Main A2C Training Algorithm
        """

        results = []

        # Main Loop
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")
        for e in tqdm_e:

            # Reset episode
            time, cumul_reward, done = 0, 0, False
            env.reset()
            old_state = env.get_observation()
            old_state = np.expand_dims(old_state, axis=0)
            actions, states, rewards = [], [], []

            while not env.is_done():

                # Actor picks an action (following the policy)
                a = self.policy_action(old_state)
                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done = env.step(a)
                # Memorize (s, a, r) for training
                action_onehot = np.zeros([self.act_dim])
                action_onehot[a] = 1
                actions.append(action_onehot)

                rewards.append(r)
                states.append(old_state)
                # Update current state
                old_state = new_state
                cumul_reward += r
                time += 1

            # Train using discounted rewards ie. compute updates
            self.train_models(states, actions, rewards, done)

            # Gather stats every episode for plotting
            # TODO

            if e % args.save_intervall == 0:
                self.save_weights(args.path)

            # Update Average Rewards
            self.av_meter.update(cumul_reward)

            # Display score
            tqdm_e.set_description("Reward Episode: " + str(cumul_reward) + " -- Averarge Reward: " + str(self.av_meter.avg))
            tqdm_e.refresh()

        return results

    def save_weights(self, path):
        path += 'A2C'
        self.actor.save_weights(path + '_actor.h5')
        self.critic.save_weights(path + '_critic.h5')

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)