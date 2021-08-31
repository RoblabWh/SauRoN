import sys

import numpy as np
from PyQt5.QtWidgets import QApplication

import EnvironmentWithUI
from BucketRenderer import BucketRenderer
from DistanceGraph import DistanceGraph
from algorithms.A2C_parallel.A2C_Multi import AverageMeter
from algorithms.A2C_parallel.PPO_Network import PPO_Network
from algorithms.A2C_parallel.PPO_MultiprocessingActor import PPO_MultiprocessingActor
from tqdm import tqdm
import ray
import yaml
#import keras
import tensorflow
import matplotlib.pyplot as plt



@ray.remote
class PPO_Multi:
    """
    Defines an Proximal Policy Optimization learning algorithm for neural nets
    """

    def __init__(self, act_dim, env_dim, args):
        """
        :param act_dim: number of available actions
        :param env_dim: (number of past states (including the current one), size of a state) -
            their product determines the number input neurons
        :param args:
        """
        self.args = args
        # self.network = PPO_Network(act_dim, env_dim, args)
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.numbOfParallelEnvs = args.parallel_envs
        self.numbOfRobots = args.numb_of_robots
        self.timePenalty = args.time_penalty
        self.av_meter = AverageMeter()
        self.gamma = args.gamma
#        ray.init()

        # self.app = QApplication(sys.argv)
        # #self.controlWindow = ControlWindowController.remote(args.parallel_envs)#, act_dim, env_dim, args)  # , model)
        # self.controlWindow = ControlWindow(args.parallel_envs, self.app)#, act_dim, env_dim, args)  # , model)
        # self.controlWindow.showandPause()

    def train(self, loadWeightsPath = ""):
        """ Main PPO Training Algorithm
        :param loadWeightsPath: The path to the .h5 file containing the pretrained weights.
         Only required if a pretrained net is used.
        """
        loadedWeights = None
        if loadWeightsPath != "":
            self.load_net(loadWeightsPath)
            loadedWeights = self.network.getWeights()
            #keras.backend.clear_session() # TODO Prüfen ob notwendig, da backend evtl. bald nicht mehr geht
            tensorflow.keras.backend.clear_session()

        #Create parallel workers with own environment
        envLevel = [(i)%4 for i in range(self.numbOfParallelEnvs)]
        #envLevel = [0 for _ in range(self.numbOfParallelEnvs)]
        #ray.init()
        multiActors = [PPO_MultiprocessingActor.remote(self.act_dim, self.env_dim, self.args, loadedWeights, envLevel[0], True)]
        startweights = multiActors[0].getWeights.remote()
        multiActors += [PPO_MultiprocessingActor.remote(self.act_dim, self.env_dim, self.args, startweights, envLevel[i+1], False) for i in range(self.numbOfParallelEnvs-1)]
        for i, actor in enumerate(multiActors):
            actor.setLevel.remote(envLevel[i])

        self.multiActors = multiActors


        # Main Loop
        tqdm_e = tqdm(range(self.args.nb_episodes), desc='Score', leave=True, unit=" episodes")

        for e in tqdm_e:
            self.currentEpisode = e

            #Start of episode for the parallel PPO actors with their own environment
            #Hier wird die gesamte Episode durchlaufen und dann erst trainiert


            #Training bereits alle n=75 steps mit den zu dem Zeitounkt gesammelten Daten (nur für noch aktive Environments)
            activeActors = multiActors
            while len(activeActors) > 0:
                futures = [actor.trainSteps.remote(self.args.train_interval) for actor in activeActors]
                allTrainingResults = ray.get(futures)
                trainedWeights = self.train_modelsFaster(allTrainingResults, multiActors[0])
                for actor in multiActors[1:len(multiActors)]:
                    actor.setWeights.remote(trainedWeights)

                activeActors = []
                for actor in multiActors:
                    if ray.get(actor.isActive.remote()):
                        activeActors.append(actor)
                print('b4 getting data')
                levelVisibilty = [True, False]#TODO CHANGE ray.get(self.controlWindow.getLevelVisibilities.remote())
                for i, show in enumerate(levelVisibilty):
                    if show and ray.get(multiActors[i].isNotShowing.remote()):
                        self.showEnvWindow(i)



            zeit, cumul_reward, done = 0, 0, False

            if (e+1) % self.args.save_intervall == 0:
                print('Saving')
                self.save_weights(multiActors[0], self.args.path)

            allReachedTargetList = []
            for actor in multiActors:
                tmpTargetList = ray.get(actor.getTargetList.remote())
                allReachedTargetList += tmpTargetList

            targetDivider = (self.numbOfParallelEnvs) * 100  # Erfolg der letzten 100
            successrate = allReachedTargetList.count(True) / targetDivider


            # Calculate and display score
            for actor in multiActors:
                (cumRewardActor, steps) = ray.get(actor.resetActor.remote())
                self.av_meter.update(cumRewardActor, steps)
                cumul_reward += cumRewardActor
            cumul_reward = cumul_reward / self.args.parallel_envs

            tqdm_e.set_description("R avr last e: " + str(cumul_reward) + " --R avr all e : " + str(self.av_meter.avg) + " --Avr Reached Target (25 epi): " + str(successrate))
            tqdm_e.refresh()

        self.save_weights(multiActors[0], self.args.path)
        for actor in multiActors:
            actor.killActor.remote()

    def prepareTraining(self, loadWeightsPath = ""):
        self.currentEpisode = 0

        loadedWeights = None
        if loadWeightsPath != "":
            self.load_net(loadWeightsPath)
            loadedWeights = self.network.getWeights()
            #keras.backend.clear_session() # TODO Prüfen ob notwendig, da backend evtl. bald nicht mehr geht
            tensorflow.keras.backend.clear_session()

        #Create parallel workers with own environment
        # envLevel = [(i)%4 for i in range(self.numbOfParallelEnvs)]
        # envLevel = [(i+3)%4 for i in range(self.numbOfParallelEnvs)]
        envLevel = [int(i/(self.numbOfParallelEnvs/5)) for i in range(self.numbOfParallelEnvs)] # TODO 4 durch anzahl der verfügbaren Level variable ersetzen
        #ray.init()
        multiActors = [PPO_MultiprocessingActor.remote(self.act_dim, self.env_dim, self.args, loadedWeights, envLevel[0], True)]
        startweights = multiActors[0].getWeights.remote()
        multiActors += [PPO_MultiprocessingActor.remote(self.act_dim, self.env_dim, self.args, startweights, envLevel[i+1], False) for i in range(self.numbOfParallelEnvs-1)]

        levelNames = []
        for i, actor in enumerate(multiActors):
            levelName = ray.get(actor.setLevel.remote(envLevel[i]))
            levelNames.append(levelName)

        self.multiActors = multiActors
        self.activeActors = multiActors


        # Main Loop
        self.tqdm_e = tqdm(range(self.args.nb_episodes), desc='Score', leave=True, unit=" episodes")

        return (False, levelNames)

    def trainWithFeedbackSteps(self, visibleLevels):

        activeActors = self.activeActors

        if len(activeActors) > 0:
            futures = [actor.trainSteps.remote(self.args.train_interval) for actor in activeActors]
            allTrainingResults = ray.get(futures)
            trainedWeights = self.train_modelsFaster(allTrainingResults, self.multiActors[0])
            for actor in self.multiActors[1:len(self.multiActors)]:
                actor.setWeights.remote(trainedWeights)

            activeActors = []
            for actor in self.multiActors:
                if ray.get(actor.isActive.remote()):
                    activeActors.append(actor)


            for i, show in enumerate(visibleLevels):
                if show:# and ray.get(self.multiActors[i].isNotShowing.remote()):
                    self.showEnvWindow(i)
                else:
                    self.hideEnvWindow(i)
            self.activeActors = activeActors

            return False
        else:
            return True

    def trainWithFeedbackEpisodes(self):
        """ Main PPO Training Algorithm
        :param loadWeightsPath: The path to the .h5 file containing the pretrained weights.
         Only required if a pretrained net is used.
        """
        if self.currentEpisode < self.args.nb_episodes:

            self.tqdm_e.update(1)
            self.currentEpisode += 1
            #Start of episode for the parallel PPO actors with their own environment
            #Hier wird die gesamte Episode durchlaufen und dann erst trainiert


            #Training bereits alle n=75 steps mit den zu dem Zeitounkt gesammelten Daten (nur für noch aktive Environments)
            self.activeActors = self.multiActors

            zeit, cumul_reward, done = 0, 0, False

            if (self.currentEpisode+1) % self.args.save_intervall == 0:
                print('Saving')
                self.save_weights(self.multiActors[0], self.args.path)

            allReachedTargetList = []
            individualSuccessrate = []
            for actor in self.multiActors:
                tmpTargetList = ray.get(actor.getTargetList.remote())
                allReachedTargetList += tmpTargetList
                individualSuccessrate.append(tmpTargetList.count(True) / 100)

            targetDivider = (self.numbOfParallelEnvs) * 100  # Erfolg der letzten 100
            successrate = allReachedTargetList.count(True) / targetDivider


            # Calculate and display score
            individualLastAverageReward = []

            for actor in self.multiActors:
                (cumRewardActor, steps) = ray.get(actor.resetActor.remote())
                self.av_meter.update(cumRewardActor, steps)
                cumul_reward += cumRewardActor
                individualLastAverageReward.append(cumRewardActor/steps)

            cumul_reward = cumul_reward / self.args.parallel_envs

            self.tqdm_e.set_description("R avr last e: " + str(cumul_reward) + " --R avr all e : " + str(self.av_meter.avg) + " --Avr Reached Target (25 epi): " + str(successrate))
            self.tqdm_e.refresh()
            return (False, individualLastAverageReward, individualSuccessrate, self.currentEpisode, successrate)
        else:
            self.save_weights(self.multiActors[0], self.args.path)
            for actor in self.multiActors:
                actor.killActor.remote()
            return (True, [], [], self.currentEpisode)



    def train_modelsFaster(self, envsData, masterEnv = None):
        statesConcatenatedL = envsData[0][0]
        statesConcatenatedO = envsData[0][1]
        statesConcatenatedD = envsData[0][2]
        statesConcatenatedV = envsData[0][3]
        statesConcatenatedT = envsData[0][4]
        discounted_rewards = envsData[0][5]
        actionsConcatenated = envsData[0][6]
        advantagesConcatenated = envsData[0][7]
        neglogsConcatinated = envsData[0][8]
        valuesConcatenated = envsData[0][9]

        for robotsData in envsData[1:]:
            statesConcatenatedL = np.concatenate((statesConcatenatedL ,robotsData[0]))
            statesConcatenatedO = np.concatenate((statesConcatenatedO ,robotsData[1]))
            statesConcatenatedD = np.concatenate((statesConcatenatedD, robotsData[2]))
            statesConcatenatedV = np.concatenate((statesConcatenatedV, robotsData[3]))
            statesConcatenatedT = np.concatenate((statesConcatenatedT, robotsData[4]))
            discounted_rewards = np.concatenate((discounted_rewards, robotsData[5]))
            actionsConcatenated = np.concatenate((actionsConcatenated, robotsData[6]))
            advantagesConcatenated = np.concatenate((advantagesConcatenated, robotsData[7]))
            neglogsConcatinated = np.concatenate((neglogsConcatinated, robotsData[8]))
            valuesConcatenated = np.concatenate((valuesConcatenated, robotsData[9]))

            i = 0
        neglogsConcatinated = np.squeeze(neglogsConcatinated)


        if masterEnv == None:
           self.network.train_net(statesConcatenatedL, statesConcatenatedO, statesConcatenatedD, statesConcatenatedV,
                                   statesConcatenatedT, discounted_rewards, actionsConcatenated, advantagesConcatenated,
                                   neglogsConcatinated, valuesConcatenated)
           weights = self.network.getWeights()
        else:
            weights = ray.get(
                masterEnv.trainNet.remote(statesConcatenatedL, statesConcatenatedO, statesConcatenatedD, statesConcatenatedV,
                                   statesConcatenatedT, discounted_rewards, actionsConcatenated, advantagesConcatenated,
                                   neglogsConcatinated, valuesConcatenated))
        return weights


    def train_models(self, envsData, masterEnv = None):#, states, actions, rewards): 1 0 2
        """
        Update actor and critic networks from experience
        :param envsData: Collected states of all robots from all used parallel environments. Collected over the last n time steps
        :param masterEnv: The environment which is used to train the network. all other networks will receive a copy
        of its weights after the training process.
        :return: trained weights of the master environments network
        """
        # Compute discounted rewards and Advantage (TD. Error)

        discounted_rewards = np.array([])
        state_values = np.array([])
        advantages = np.array([])
        actionsConcatenated = np.array([])
        statesConcatenatedL = np.array([])
        statesConcatenatedO = np.array([])
        statesConcatenatedD = np.array([])
        statesConcatenatedV = np.array([])
        statesConcatenatedT = np.array([])
        neglogsConcatinated = np.array([])
        for robotsData in envsData:
            for data in robotsData:
                actions, states, rewards, dones, evaluations, neglogs = data

                if (actionsConcatenated.size == 0):
                    actionsConcatenated = np.vstack(actions)

                else:
                    actionsConcatenated = np.concatenate((actionsConcatenated, np.vstack(actions)))


                lasers = []
                orientations = []
                distances = []
                velocities = []
                usedTimeSteps = []

                for s in states:
                    laser = np.array([np.array(s[i][0]) for i in range(0, len(s))]).swapaxes(0,1)
                    orientation = np.array([np.array(s[i][1]) for i in range(0, len(s))]).swapaxes(0,1)
                    distance = np.array([np.array(s[i][2]) for i in range(0, len(s))]).swapaxes(0,1)
                    velocity = np.array([np.array(s[i][3]) for i in range(0, len(s))]).swapaxes(0,1)
                    usedTimeStep = np.array([np.array(s[i][4]) for i in range(0, len(s))])

                    lasers.append(laser)
                    orientations.append(orientation)
                    distances.append(distance)
                    velocities.append(velocity)
                    usedTimeSteps.append(usedTimeStep)

                if(statesConcatenatedL.size == 0):
                    statesConcatenatedL = np.array(lasers)
                    statesConcatenatedO = np.array(orientations)
                    statesConcatenatedD = np.array(distances)
                    statesConcatenatedV = np.array(velocities)
                    statesConcatenatedT = np.array(usedTimeSteps)
                    state_values = np.array(evaluations)
                    neglogsConcatinated = np.array(neglogs)
                else:
                    statesConcatenatedL = np.concatenate((statesConcatenatedL, np.array(lasers)))
                    statesConcatenatedO = np.concatenate((statesConcatenatedO, np.array(orientations)))
                    statesConcatenatedD = np.concatenate((statesConcatenatedD, np.array(distances)))
                    statesConcatenatedV = np.concatenate((statesConcatenatedV, np.array(velocities)))
                    statesConcatenatedT = np.concatenate((statesConcatenatedT, np.array(usedTimeSteps)))
                    state_values = np.concatenate((state_values, evaluations))
                    neglogsConcatinated = np.concatenate((neglogsConcatinated, np.array(neglogs)))

                discounted_rewardsTmp = self.discount(rewards)
                discounted_rewards = np.concatenate((discounted_rewards, discounted_rewardsTmp))



                advantagesTmp = discounted_rewardsTmp - np.reshape(evaluations, len(evaluations))  # Warum reshape
                advantagesTmp = (advantagesTmp - advantagesTmp.mean()) / (advantagesTmp.std() + 1e-8)
                advantages = np.concatenate((advantages, advantagesTmp))


                # print("discounted_rewards", discounted_rewards.shape, "state_values", state_values.shape, "advantages",
                #       advantages.shape, "actionsConcatenated", actionsConcatenated.shape, np.vstack(actions).shape)
                # print(len(statesConcatenatedL), len(statesConcatenatedO), len(statesConcatenatedD), len(statesConcatenatedV), len(discounted_rewards), len(actionsConcatenated), len(advantages))

        neglogsConcatinated = np.squeeze(neglogsConcatinated)
        if masterEnv == None:
            #for i in len(statesConcatenatedL):
             #   self.network.train_net(statesConcatenatedL[i], statesConcatenatedO[i], statesConcatenatedD[i],statesConcatenatedV[i], statesConcatenatedT[i],discounted_rewards[i], actionsConcatenated[i],advantages[i], neglogsConcatinated[i])
            self.network.train_net(statesConcatenatedL, statesConcatenatedO, statesConcatenatedD,statesConcatenatedV, statesConcatenatedT,discounted_rewards, actionsConcatenated,advantages, neglogsConcatinated)
            weights = self.network.getWeights()
        else:
            weights, var = ray.get(masterEnv.trainNet.remote(statesConcatenatedL, statesConcatenatedO, statesConcatenatedD,statesConcatenatedV, statesConcatenatedT,discounted_rewards, actionsConcatenated,advantages, neglogsConcatinated))
        return weights , var

    def discount(self, r):
        """
        Compute the gamma-discounted rewards over an episode
        """
        discounted_r = np.zeros_like(r, dtype=float)
        cumul_r = 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def saveCurrentWeights(self):
        """
        saves weights at current epoch
        """
        print('Saving individual')
        self.save_weights(self.args.path, "_e" + str(self.currentEpisode))

    def save_weights(self, masterEnv, path, additional=""):
        path += 'PPO' + self.args.model_timestamp + additional

        masterEnv.saveWeights.remote(path)
        data = [self.args]
        with open(path+'.yml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

    def load_net(self, path):
        self.network = PPO_Network(self.act_dim, self.env_dim, self.args)
        self.network.load_weights(path)
        return True


    def showEnvWindow(self, envID):
        if envID < len(self.multiActors):
            self.multiActors[envID].showWindow.remote()

    def hideEnvWindow(self, envID):
        if envID < len(self.multiActors):
            self.multiActors[envID].hideWindow.remote()

    def execute(self, args, env_dim):
        """
        executes a trained net (without using the standard deviation in the action selection)
        :param env: EnvironmentWithUI.Environment
        :param args: args defined in main
        """
        print("hello there")
        app = QApplication(sys.argv)
        env = EnvironmentWithUI.Environment(app, args, env_dim, 0)
        env.simulation.showWindow(app)



        # visualization of chosen actions
        #histogramm = BucketRenderer(20, 0)
        #histogramm.show()
        liveHistogramRobot = 0

        #robotsCount = self.numbOfRobots #TODO bei jedem neuen Level laden akualisieren

        distGraph = DistanceGraph(app)
        for e in range(18):
            env.reset(0)#e % len(env.simulation.levelFiles))
            robotsCount = env.simulation.getCurrentNumberOfRobots()
            robotsOldState = [np.expand_dims(env.get_observation(i), axis=0) for i in range(0, robotsCount)]
            robotsDone = [False for i in range(0, robotsCount)]

            while not env.is_done():
                robotsActions = []
                robotsHeatmaps = []
                # Actor picks an action (following the policy)
                for i in range(0, robotsCount):
                    if not robotsDone[i]:
                        aTmp, heatmap = self.network.policy_action_certain(robotsOldState[i][0])
                        a = np.ndarray.tolist(aTmp[0].numpy())

                        if i == liveHistogramRobot:
                            if heatmap != None:
                                distGraph.plot([i for i in range(heatmap.shape[0])], heatmap)
                            # heatmap = np.maximum(heatmap, 0)
                            # heatmap /= np.max(heatmap)
                            # print(heatmap)
                            # plt.plot(heatmap)
                            # # matshow(heatmap)
                            # plt.show()

                        #visualization of chosen actions
                        #if i == liveHistogramRobot:
                            #histogramm.add_action(a)
                            #histogramm.show()

                    else:
                        a = [None, None]
                        heatmap = None
                    robotsActions.append(a[0])
                    robotsHeatmaps.append(heatmap)
                if args.lidar_activation:
                    robotsStates = env.step(robotsActions, robotsHeatmaps)
                else:
                    robotsStates = env.step(robotsActions)

                rewards = ''
                for i, stateData in enumerate(robotsStates):
                    new_state = stateData[0]
                    rewards += (str(i) + ': ' + str(stateData[1]) + '   ')
                    done = stateData[2]
                    robotsOldState[i] = new_state
                    if not robotsDone[i]:
                        robotsDone[i] = done
