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
        self.args = args

        # self.a_opt = self.actor.optimizer()
        # self.c_opt = self.critic.optimizer()

    def buildNetwork(self):
        """ Assemble shared layers
        """
        inp = Input(self.env_dim)
        x = Flatten()(inp)
        x = Dense(128, activation='relu')(x) #64
        x = Dense(256, activation='relu')(x) #128
        return Model(inp, x)

    def buildActor(self, network):
        x = Dense(64, activation='relu')(network.output) #128
        out = Dense(self.act_dim, activation='tanh')(x)
        #TODO hier muss eine Ausgabelayer mit 2 Werten je zwischen -1 und 1 entstehen
        return Model(network.input, out)

    def buildCritic(self, network):
        x = Dense(64, activation='relu')(network.output) #128
        out = Dense(1, activation='linear')(x)
        return Model(network.input, out)

    def policy_action(self, s, successrate):
        """ Use the actor to predict the next action to take, using the policy
        """
        #print(self.actor.predict(s).ravel())
        std = ((1-successrate)**2)*0.8
        prediction = self.actor.predict(s).ravel()
        # print("prediction: " + str(prediction))
        prediction[0] = np.random.normal(prediction[0], std)
        prediction[1] = np.random.normal(prediction[1], std)
        # print("prediction: " + str(prediction) + "standardabweichung: " + str(std) + "  - clipped: " + str(np.clip(prediction, -1, 1)))
        return np.clip(prediction, -1, 1)
        #return np.random.choice(np.arange(self.act_dim), 1, p=self.actor.predict(s).ravel())[0]

    def discount(self, r):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r = np.zeros_like(r, dtype=float)
        cumul_r = 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def train_models(self, states, actions, rewards):
        """ Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(rewards)
        # print(states.shape)
        # states = np.vstack(states)
        # print(states.shape)

        state_values = self.critic.predict(np.asarray(states))[:,0]
        advantages = discounted_rewards - np.reshape(state_values, len(state_values))  # Warum reshape
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Networks optimization
        # self.a_opt([states, actions, advantages])
        # self.c_opt([states, discounted_rewards])
        actions = np.vstack(actions)
        self.actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        self.critic.fit(states, discounted_rewards, epochs=1, verbose=0)

    def train(self, env, args):
        """ Main A2C Training Algorithm
        """

        results = []            # wird nirgendwo gebraucht -> returned leeres Array
        counter = 1
        liste = np.array([], dtype=object)
        # Main Loop
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")
        waitForN = 10
        rechedTargetList = [False] * 100
        for e in tqdm_e:

            # Reset episode
            time, cumul_reward, done = 0, 0, False
            env.reset()


            #TODO irgendwo anders her bekommen (zentral)
            countRobots = 4


            robotsData = []
            robotsOldState = []


            for i in range(countRobots):
                old_state = env.get_observation(i)
                robotsOldState.append(np.expand_dims(old_state, axis=0))


                actions, states, rewards, done = [], [], [], []
                robotsData.append((actions, states, rewards, done))
            # Robot 0 actions --> robotsData[0][0]
            # Robot 0 states  --> robotsData[0][1]
            # Robot 0 rewards --> robotsData[0][2]
            # Robot 1 actions --> robotsData[1][0]
            # ...


            while not env.is_done():


                robotsActions = []
                # Actor picks an action (following the policy)
                for i in range(0,len(robotsData)):
                    if not True in robotsData[i][3]:
                        a = self.policy_action(robotsOldState[i], sum(rechedTargetList)/100)
                        # print(a)
                    else:
                        a = [None, None]
                    robotsActions.append(a)
                    # action_onehot = np.zeros([self.act_dim])
                    # action_onehot[a] = 1

                    if not None in a:
                        robotsData[i][0].append(a)#action_onehot) #TODO Tupel mit 2 werten von je -1 bis 1


                # Retrieve new state, reward, and whether the state is terminal
                # new_state, r, done = env.step(robotsActions)

                robotsDataCurrentFrame = env.step(robotsActions)

                #print("reward " + str(r))
                # Memorize (s, a, r) for training

                for i, dataCurrentFrame in enumerate(robotsDataCurrentFrame):

                    if not True in robotsData[i][3]:
                        new_state = dataCurrentFrame[0]
                        r = dataCurrentFrame[1]
                        done = dataCurrentFrame[2]
                        robotsData[i][1].append(old_state)
                        robotsData[i][2].append(r)
                        robotsData[i][3].append(done)
                        if(done):
                            rechedPickup = dataCurrentFrame[3]
                            rechedTargetList.pop(0)
                            rechedTargetList.append(rechedPickup)
                        # Update current state
                        robotsOldState[i] = new_state
                        #print(r)
                        cumul_reward += r
                #print("Kumulierter Reward: " + str(cumul_reward) + ", Reward: " + str(r))
                time += 1


            # Train using discounted rewards ie. compute updates
            # liste = np.append([liste], [[states], [actions], [rewards], [done]])
            #
            #
            # if counter == waitForN:   # train after 9 Episodes
            #     for i in range(0, liste.size, 4):
            #         self.train_models(liste[i+0], liste[i+1], liste[i+2], liste[i+3])
            #
            #     liste = np.array([], dtype=object)
            #     counter = 0
            #
            # counter += 1
            # Gather stats every episode for plotting

            for singleRobotData in robotsData:
                # print(singleRobotData[1], singleRobotData[0], singleRobotData[2])
                self.train_models(np.asarray(singleRobotData[1]), singleRobotData[0], singleRobotData[2])


            if e % args.save_intervall == 0:
                print('Saving')
                self.save_weights(args.path)

            # Update Average Rewards
            self.av_meter.update(cumul_reward)

            # Display score
            tqdm_e.set_description("Reward Episode: " + str(cumul_reward) + " -- Average Reward: " + str(self.av_meter.avg) + " Average Reached Target (last 100): " + str(sum(rechedTargetList)/100))
            tqdm_e.refresh()

        return results

    def save_weights(self, path):
        path += 'A2C'
        self.actor.save_weights(path + '_actor_' + self.args.mode + '.h5')
        self.critic.save_weights(path + '_critic_' + self.args.mode + '.h5')

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)

    def execute(self, env, args):
        state = env.get_observation()
        state = np.expand_dims(state, axis=0)

        while not env.is_done():
            new_state, r, done = env.step(np.argmax(self.actor.predict(state).ravel()))
            #print(np.argmax(self.actor.predict(state).ravel()), self.actor.predict(state).ravel(), self.actor.predict(state))
            state = new_state