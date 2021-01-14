import numpy as np
import keras as k
import copy
import concurrent.futures

from tqdm import tqdm
from algorithms.A2C_Network import A2C_Network
import ray


from algorithms.A2C_NetworkCopy import A2C_NetworkCopy
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Lambda, Conv1D, concatenate
from keras.optimizers import RMSprop, Adam
from keras.losses import mean_squared_error
from keras.layers import Input, Conv1D, Dense, Flatten, concatenate, MaxPool1D, Lambda
from keras.backend import max, mean, exp, log, function, squeeze, categorical_crossentropy,placeholder, sum, square, random_normal, shape, cast, clip, softmax, argmax
from keras import backend as K

from utils import AverageMeter


@ray.remote
def trainSingleEnv(network, args, env):
    import tensorflow as tf
    """ Main A2C Training Algorithm
    """
    #Todo action und env Dimensions nicht manuell setzen
    # network = A2C_NetworkCopy(2, (4,187), args, weights)
    rechedTargetList = [False] * 100
    countRobots = 1  # TODO aus environment bekommen

    # Reset episode
    time, cumul_reward, done = 0, 0, False
    env.reset()

    robotsData = []
    robotsOldState = []
    for i in range(countRobots):
        old_state = env.get_observation(i)
        robotsOldState.append(np.expand_dims(old_state, axis=0))


        actions, states, rewards, done, evaluation = [], [], [], [], []
        robotsData.append((actions, states, rewards, done, evaluation))
        # Robot 0 actions --> robotsData[0][0]
        # Robot 0 states  --> robotsData[0][1]
        # Robot 0 rewards --> robotsData[0][2]
        # Robot 1 actions --> robotsData[1][0]
        # ...

    while not env.is_done():

        robotsActions = []
        # Actor picks an action (following the policy)
        for i in range(0, len(robotsData)):
            if not True in robotsData[i][3]:
                aTmp = network.policy_action(robotsOldState[i][0], (rechedTargetList).count(True) / 100)
                a = np.ndarray.tolist(aTmp[0])[0]
                c = np.ndarray.tolist(aTmp[1])[0]
            else:
                a = [None, None]
            robotsActions.append(a)

            if not None in a:
                robotsData[i][0].append(a)
                robotsData[i][4].append(c)

        # Retrieve new state, reward, and whether the state is terminal
        robotsDataCurrentFrame = env.step(robotsActions)

        # Memorize (s, a, r) for training
        for i, dataCurrentFrame in enumerate(robotsDataCurrentFrame):

            if not True in robotsData[i][3]:
                new_state = dataCurrentFrame[0]
                r = dataCurrentFrame[1]
                done = dataCurrentFrame[2]
                robotsData[i][1].append(robotsOldState[i][0])
                robotsData[i][2].append(r)
                robotsData[i][3].append(done)
                if (done):
                    reachedPickup = dataCurrentFrame[3]
                    rechedTargetList.pop(0)
                    rechedTargetList.append(reachedPickup)
                # Update current state
                robotsOldState[i] = new_state
                cumul_reward += r
        # print("Kumulierter Reward: " + str(cumul_reward) + ", Reward: " + str(r))
        time += 1

    return robotsData

class A2C_C:
    """ Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, args):
        """ Initialization
        """
        print(k.__version__)
        ray.init()
        # # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = env_dim
        # self.gamma = args.gamma
        # self.lr = args.learningrate
        # self.rays = int(360/args.angle_steps)
        # self.timePenalty = args.time_penalty
        #
        # self._input_laser = Input(shape=(4, self.rays), dtype='float32', name='input_laser')
        # # Orientation input
        # self._input_orientation = Input(shape=(4, 2,), dtype='float32', name='input_orientation')
        # # Distance input
        # self._input_distance = Input(shape=(4, 1,), dtype='float32', name='input_distance')
        # # Velocity input
        # self._input_velocity = Input(shape=(4, 2,), dtype='float32', name='input_velocity')
        # # Passed Time input
        # self._input_timestep = Input(shape=(1, 1,), dtype='float32', name='input_Timestep')
        #
        # # Create actor and critic networks
        # self.buildNetWithOpti()
        #
        # self.av_meter = AverageMeter()
        self.network = A2C_Network(act_dim, env_dim, args)
        self.args = args

    def policy_action_certain(self, s):#TODO obs_timestep mit übergeben
        """ Use the actor to predict the next action to take, using the policy
        """
        # std = ((1-successrate)**2)*0.55


        laser = np.array([np.array(s[i][0]) for i in range(0, len(s))])
        orientation = np.array([np.array(s[i][1]) for i in range(0, len(s))])
        distance = np.array([np.array(s[i][2]) for i in range(0, len(s))])
        velocity = np.array([np.array(s[i][3]) for i in range(0, len(s))])
        # timesteps = np.array([np.array(0) for i in range(0, len(s))])
        timesteps = np.array([np.array(s[i][4]) for i in range(0, len(s))])
        if(self.timePenalty):
            mu = self._sample([np.array([laser]),  np.array([orientation]), np.array([distance]), np.array([velocity]), np.array([timesteps])])
        else:
            mu = self._sample([np.array([laser]),  np.array([orientation]), np.array([distance]), np.array([velocity])])

        return mu


    def trainMultiple(self, envs, args):

        nbrOfCopies = 2  # TODO zentral über args
        # envs = [copy.copy(env) for _ in range(nbrOfCopies)]
        weights = [self.network.getWeights() for _ in range(nbrOfCopies)]

        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")
        for e in tqdm_e:
            networks = [copy.copy(self.network) for _ in range(nbrOfCopies)]
            argsForAll = [args for _ in range(nbrOfCopies)]
            futures = [trainSingleEnv.remote(networks[i], args, envs[i]) for i in range(nbrOfCopies)]
            results = ray.get(futures)
            # TODO Listengebastel muss hier hin
            for r in results:
                print(str(e))
            #self.trainModelsAllRobots(results)

            # if e % args.save_intervall == 0:
            #     print('Saving')
            #     self.save_weights(args.path)
            #
            # # Update Average Rewards
            # self.av_meter.update(cumul_reward)

            # Display score
            # tqdm_e.set_description("Reward Episode: " + str(cumul_reward) + " -- Average Reward: " + str(
            #     self.av_meter.avg) + " Average Reached Target (last 100): " + str(
            #     (rechedTargetList).count(True) / 100))

            tqdm_e.set_description("done with 4 more (total episodes run: )"+str(e))
            tqdm_e.refresh()




    # def train(self, envs, args):
    #     """ Main A2C Training Algorithm
    #     """
    #
    #     # Main Loop
    #     tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")
    #     waitForN = 10
    #     rechedTargetList = [False] * 100
    #     countRobots = 1
    #
    #     for e in tqdm_e:
    #
    #         # Reset episode
    #         time, cumul_reward, done = 0, 0, False
    #
    #         allEnvRobotsDatas = []
    #         allEnvOldStates = []
    #         for j, env in enumerate(envs):
    #
    #             env.reset()
    #             robotsData = []
    #             robotsOldState = []
    #
    #             for i in range(countRobots):
    #                 old_state = env.get_observation(i)
    #                 robotsOldState.append(np.expand_dims(old_state, axis=0))
    #
    #                 actions, states, rewards, done, evaluation = [], [], [], [], []
    #                 robotsData.append((actions, states, rewards, done, evaluation))
    #                 # Robot 0 actions --> robotsData[0][0]
    #                 # Robot 0 states  --> robotsData[0][1]
    #                 # Robot 0 rewards --> robotsData[0][2]
    #                 # Robot 1 actions --> robotsData[1][0]
    #                 # ...
    #             allEnvRobotsDatas[j].append(robotsData)
    #             allEnvOldStates[j].append(robotsOldState)
    #
    #         while not env.is_done():
    #
    #             robotsActions = []
    #             # Actor picks an action (following the policy)
    #             for i in range(0, len(robotsData)):
    #                 if not True in robotsData[i][3]:
    #                     # a = self.predict(robotsOldState[i][0:90][:], )
    #                     # TODO vielleicht Zeit nehmen von policy action
    #                     aTmp = self.policy_action(robotsOldState[i][0], (rechedTargetList).count(True) / 100)
    #                     a = np.ndarray.tolist(aTmp[0])[0]
    #                     c = np.ndarray.tolist(aTmp[1])[0]
    #                     # print(a,c)
    #                 else:
    #                     a = [None, None]
    #                 robotsActions.append(a)
    #                 # action_onehot = np.zeros([self.act_dim])
    #                 # action_onehot[a] = 1
    #
    #                 if not None in a:
    #                     robotsData[i][0].append(a)  # action_onehot) #TODO Tupel mit 2 werten von je -1 bis 1
    #                     robotsData[i][4].append(c)
    #
    #             # Retrieve new state, reward, and whether the state is terminal
    #             # new_state, r, done = env.step(robotsActions)
    #             # TODO time hier und danach nehmen und dann die Differenz angucken, einmal für step und einmal für den Rest
    #             robotsDataCurrentFrame = env.step(robotsActions)
    #
    #             # print("reward " + str(r))
    #             # Memorize (s, a, r) for training
    #
    #             for i, dataCurrentFrame in enumerate(robotsDataCurrentFrame):
    #
    #                 if not True in robotsData[i][3]:
    #                     new_state = dataCurrentFrame[0]
    #                     r = dataCurrentFrame[1]
    #                     done = dataCurrentFrame[2]
    #                     robotsData[i][1].append(robotsOldState[i][0])
    #                     robotsData[i][2].append(r)
    #                     robotsData[i][3].append(done)
    #                     if (done):
    #                         reachedPickup = dataCurrentFrame[3]
    #                         rechedTargetList.pop(0)
    #                         rechedTargetList.append(reachedPickup)
    #                     # Update current state
    #                     robotsOldState[i] = new_state
    #                     cumul_reward += r
    #             # print("Kumulierter Reward: " + str(cumul_reward) + ", Reward: " + str(r))
    #             time += 1
    #
    #         self.train_models(robotsData)
    #
    #         if e % args.save_intervall == 0:
    #             print('Saving')
    #             self.save_weights(args.path)
    #
    #         # Update Average Rewards
    #         self.av_meter.update(cumul_reward)
    #
    #         # Display score
    #         tqdm_e.set_description("Reward Episode: " + str(cumul_reward) + " -- Average Reward: " + str(
    #             self.av_meter.avg) + " Average Reached Target (last 100): " + str((rechedTargetList).count(True) / 100))
    #         tqdm_e.refresh()
    #
    #     return results

    def train(self, env, args):
        """ Main A2C Training Algorithm
        """

        results = []  # wird nirgendwo gebraucht -> returned leeres Array

        liste = np.array([], dtype=object)
        # Main Loop
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")
        waitForN = 10
        rechedTargetList = [False] * 100
        countRobots = args.nb_robots

        for e in tqdm_e:

            # Reset episode
            time, cumul_reward, done = 0, 0, False
            env.reset()

            # TODO irgendwo anders her bekommen (zentral)

            robotsData = []
            robotsOldState = []

            for i in range(countRobots):
                old_state = env.get_observation(i)
                robotsOldState.append(np.expand_dims(old_state, axis=0))

                actions, states, rewards, done, evaluation = [], [], [], [], []
                robotsData.append((actions, states, rewards, done, evaluation))
            # Robot 0 actions --> robotsData[0][0]
            # Robot 0 states  --> robotsData[0][1]
            # Robot 0 rewards --> robotsData[0][2]
            # Robot 1 actions --> robotsData[1][0]
            # ...

            while not env.is_done():

                robotsActions = []
                # Actor picks an action (following the policy)
                for i in range(0, len(robotsData)):
                    if not True in robotsData[i][3]:
                        # TODO vielleicht Zeit nehmen von policy action
                        aTmp = self.network.policy_action(robotsOldState[i][0], (rechedTargetList).count(True) / 100)
                        a = np.ndarray.tolist(aTmp[0])[0]
                        c = np.ndarray.tolist(aTmp[1])[0]
                        # print(a,c)
                    else:
                        a = [None, None]
                    robotsActions.append(a)

                    if not None in a:
                        robotsData[i][0].append(a)  # action_onehot) #TODO Tupel mit 2 werten von je -1 bis 1
                        robotsData[i][4].append(c)

                # Retrieve new state, reward, and whether the state is terminal
                # new_state, r, done = env.step(robotsActions)
                # TODO time hier und danach nehmen und dann die Differenz angucken, einmal für step und einmal für den Rest
                robotsDataCurrentFrame = env.step(robotsActions)

                # print("reward " + str(r))
                # Memorize (s, a, r) for training

                for i, dataCurrentFrame in enumerate(robotsDataCurrentFrame):

                    if not True in robotsData[i][3]:
                        new_state = dataCurrentFrame[0]
                        r = dataCurrentFrame[1]
                        done = dataCurrentFrame[2]
                        robotsData[i][1].append(robotsOldState[i][0])
                        robotsData[i][2].append(r)
                        robotsData[i][3].append(done)
                        if (done):
                            reachedPickup = dataCurrentFrame[3]
                            rechedTargetList.pop(0)
                            rechedTargetList.append(reachedPickup)
                        # Update current state
                        robotsOldState[i] = new_state
                        cumul_reward += r
                # print("Kumulierter Reward: " + str(cumul_reward) + ", Reward: " + str(r))
                time += 1

            self.network.train_models(robotsData)

            if e % args.save_intervall == 0:
                print('Saving')
                #self.save_weights(args.path)

            # Update Average Rewards
            #self.av_meter.update(cumul_reward)

            # Display score
            tqdm_e.set_description("Reward Episode: " + str(cumul_reward) + " -- Average Reward: " + str(2
                ) + " Average Reached Target (last 100): " + str((rechedTargetList).count(True) / 100)) #self.av_meter.avg
            tqdm_e.refresh()

        return results


    def save_weights(self, path):
        path += 'A2C'
        self._model.save_weights(path + '_actor_Critic_' + self.args.mode + '.h5')

    def load_weights(self, path):
        self._model.load_weights(path)

   # def load_weights(self, path_actor, path_critic):
   #     self.critic.load_weights(path_critic)
   #     self.actor.load_weights(path_actor)

    #TODO execute für jedes einzelne environment aufrufen und am ende Daten sammeln und ins training werfen

    def execute(self, env, args):
        robotsCount = 4

        for e in range (0,4):

            env.reset()
            #TODO nach erstem Mal auf trainiertem env sowas wie environment.randomizeForTesting() einbauen (alternativ hier ein fester Testsatz)
            # robotsOldState = [env.get_observation(i) for i in range(0, robotsCount)]
            robotsOldState = [np.expand_dims(env.get_observation(i), axis=0) for i in range(0, robotsCount)]
            robotsDone = [False for i in range(0, robotsCount)]

            while not env.is_done():

                robotsActions = []
                # Actor picks an action (following the policy)
                for i in range(0, robotsCount):

                    if not robotsDone[i]:
                        aTmp = self.policy_action_certain(robotsOldState[i][0])#, (rechedTargetList).count(True) / 100)
                        a = np.ndarray.tolist(aTmp[0])[0]
                    else:
                        a = [None, None]

                    robotsActions.append(a)

                #Das hier für jedes Environment
                robotsStates = env.step(robotsActions)

                rewards = ''
                for i, stateData in enumerate(robotsStates):
                    new_state = stateData[0]
                    rewards += (str(i)+': '+str(stateData[1]) + '   ')
                    done = stateData[2]

                    # if (done):
                    #     reachedPickup = stateData[3]
                    #     rechedTargetList.pop(0)
                    #     rechedTargetList.append(reachedPickup)

                    robotsOldState[i] = new_state
                    if not robotsDone[i]:
                        robotsDone[i]= done
                print(rewards)


