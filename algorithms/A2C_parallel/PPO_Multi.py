import sys

import numpy as np
from PyQt5.QtWidgets import QApplication

import EnvironmentWithUI
from BucketRenderer import BucketRenderer
from DistanceGraph import DistanceGraph
from algorithms.A2C_parallel.A2C_Multi import AverageMeter
#from algorithms.A2C_parallel.PPO_Network import PPO_Network
from algorithms.A2C_parallel.PPO_Network_NewContinuousLayer import PPO_Network
from algorithms.A2C_parallel.PPO_MultiprocessingActor import PPO_MultiprocessingActor
from algorithms.A2C_parallel.robins.A2C_Network_robin import Robin_Network
from tqdm import tqdm
import ray
import yaml
import tensorflow
import matplotlib.pyplot as plt
import time
from random import shuffle


import sys
# insert at 1, 0 is the script path (or '' in REPL)

sys.path.insert(1, '/')
# sys.path.insert(1, 'C:/Users/Jenny/Downloads/aia-trt-inference-master/aia-trt-inference-master/fuzzy_controller')
from displayWidget import DisplayWidget



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
        self.levelFiles = args.level_files
        # self.network = PPO_Network(act_dim, env_dim, args)
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.numbOfParallelEnvs = args.parallel_envs
        self.numbOfRobots = args.numb_of_robots
        self.timePenalty = args.time_penalty
        self.av_meter = AverageMeter()
        self.gamma = args.gamma


    def prepare_training(self, loadWeightsPath = ""):
        self.currentEpisode = 0

        loadedWeights = None
        if loadWeightsPath != "":
            self.load_net(loadWeightsPath)
            #loadedWeights = self.network.getWeights()
            loadedWeights = self.network.get_model_weights() #Robins
            #keras.backend.clear_session() # TODO Prüfen ob notwendig, da backend evtl. bald nicht mehr geht
            tensorflow.keras.backend.clear_session()

        #Create parallel workers with own environment
        # envLevel = [(i)%4 for i in range(self.numbOfParallelEnvs)]
        # envLevel = [(i+3)%4 for i in range(self.numbOfParallelEnvs)]
        envLevel = [int(i/(self.numbOfParallelEnvs/len(self.levelFiles))) for i in range(self.numbOfParallelEnvs)]
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


    def train_with_feedback_for_n_steps(self, visibleLevels):

        activeActors = self.activeActors

        if len(activeActors) > 0:
            futures = [actor.trainSteps.remote(self.args.train_interval) for actor in activeActors]
            allTrainingResults = ray.get(futures) #Liste mit Listen von Observations aus den Environments
            # trainedWeights = self.train_modelsFaster(allTrainingResults, self.multiActors[0])
            # start = time.time()
            trainedWeights = self.train_models_with_obs(allTrainingResults, self.multiActors[0])
            # end = time.time()
            # print(' time used for training: ', end-start)
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


    def train_with_feedback_end_of_episode(self):
        """
        Called after a full episode of training to collect statistics, reset the active actors list and save weights
        """
        if self.currentEpisode < self.args.nb_episodes:

            self.tqdm_e.update(1)
            self.currentEpisode += 1
            #reset the active actors list for next episode
            self.activeActors = self.multiActors

            #save weights in predefined interval
            if (self.currentEpisode+1) % self.args.save_intervall == 0:
                print('Saving')
                self.save_weights(self.multiActors[0], self.args.path)

            #build statistics of last episode
            zeit, cumul_reward, done = 0, 0, False

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
            #If all episodes are finished the current weights are saved
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


    def train_models_with_obs(self, obs_lists, master_env):
        shuffle(obs_lists)
        numb_of_exp_per_batch = int(2400 / (self.args.train_interval  * 4))

        obs_concatinated = []
        current_index = -1

        weights = None
        for i, exp in enumerate(obs_lists):
            if(i%numb_of_exp_per_batch == 0):
                current_index += 1
                obs_concatinated.append(exp)
            else:
                # Print structure of dict
                # for key, value in obs_concatinated[current_index].items():
                #     print(key, type(value))
                #     if isinstance(value, dict):
                #         for key2, value2 in value.items():
                #             print("└─→", key2, type(value2))
                #
                # for key, value in exp.items():
                #     print(key, type(value))
                #     if isinstance(value, dict):
                #         for key2, value2 in value.items():
                #             print("└─→", key2, type(value2))


                # print((obs_concatinated[current_index].items()))
                for key, value in obs_concatinated[current_index].items():
                    if type(value) == type(exp):
                        for key2, value2 in value.items():
                            obs_concatinated[current_index][key][key2] = np.concatenate((value2, exp[key][key2]))
                    else:
                        obs_concatinated[current_index][key] = np.concatenate((value, exp[key]))


        for exp in obs_concatinated:
            weights = ray.get(master_env.train_net_obs.remote(exp))
        return weights


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
        # self.network = PPO_Network(self.act_dim, self.env_dim, self.args)
        # self.network.load_weights(path)
        self.network = Robin_Network(self.act_dim, self.env_dim, self.args)
        self.network.build()
        self.network.load_weights(path)
        #self.network.load_model(path)
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

        #auskommentieren wenn fuzzy nicht genutzt wird
        # displayWidget = DisplayWidget(9)#9)
        # displayWidget.show()

        # visualization of chosen actions
        #histogramm = BucketRenderer(20, 0)
        #histogramm.show()
        #iveHistogramRobot = 0

        #robotsCount = self.numbOfRobots #TODO bei jedem neuen Level laden akualisieren

        distGraph = DistanceGraph(app)
        for e in range(18):
            env.reset(e % len(env.simulation.levelFiles))
            robotsCount = env.simulation.getCurrentNumberOfRobots()
            robotsOldState = [np.expand_dims(env.get_observation(i), axis=0) for i in range(0, robotsCount)]
            robotsDone = [False for i in range(0, robotsCount)]

            while not env.is_done():
                robotsActions = []
                robotsHeatmaps = []
                # Actor picks an action (following the policy)
                for i in range(0, robotsCount):
                    if not robotsDone[i]:
                        # aTmp, heatmap = self.network.policy_action_certain(robotsOldState[i][0]) #i for selected robot, 0 beause the state is encapsulated once too much
                        aTmp, heatmap = self.network.pedict_certain(robotsOldState[i][0]) #robin
                        a = np.ndarray.tolist(aTmp[0].numpy())

                        # if i == liveHistogramRobot:
                            # distGraph.plot([i for i in range(heatmap.shape[0])], heatmap)
                            # heatmap = np.maximum(heatmap, 0)
                            # heatmap /= np.max(heatmap)
                            # print(heatmap)
                            # plt.plot(heatmap)
                            # # matshow(heatmap)
                            # plt.show()

                        #FUZZY
                        #robotsFuzzy = displayWidget.getRobots()
                        # if i in robotsFuzzy:# and args.use_fuzzy:
                        #     obs = []
                        #     for e_len in range(len(robotsOldState[i][0][3]) - 1): #aktuelle Frame des states liegt auf 3 nicht 0
                        #         entry = robotsOldState[i][0][3][e_len]
                        #         sh = np.asarray(entry)
                        #         sh = sh.reshape((1, len(sh), 1))
                        #         obs.append(sh)
                        #     obs = (obs[0], obs[1], obs[2], obs[3]) #[lidarDistances, orientationToGoal, normaizedDistance, [linVel, angVel]]   without currentTimestep
                        #     displayWidget.setRobot(i)
                        #     aCur = displayWidget.step(obs, aTmp)
                        #
                        #     if displayWidget.aggregated:
                        #         a = aCur[0]

                        #visualization of chosen actions
                        #if i == liveHistogramRobot:
                            #histogramm.add_action(a)
                            #histogramm.show()

                    else:
                        a = [None, None]
                        heatmap = None
                    robotsActions.append(a)
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


    def trainPerception(self, args, env_dim):
        """
        :param env: EnvironmentWithUI.Environment
        :param args: args defined in main
        """
        app = QApplication(sys.argv)
        env = EnvironmentWithUI.Environment(app, args, env_dim, 0)
        env.simulation.showWindow(app)
        self.network.create_perception_model()

        inspectedRobot = 0

        traingDataStates = []
        traingDataProximityCategories = []

        for e in range(10):
            env.reset(e % len(env.simulation.levelFiles))
            robotsCount = env.simulation.getCurrentNumberOfRobots()
            robotsOldState = [np.expand_dims(env.get_observation(i), axis=0) for i in range(0, robotsCount)]
            robotsDone = [False for i in range(0, robotsCount)]

            while not env.is_done():
                robotsActions = []
                # Actor picks an action (following the policy)
                proximityCategory = 0
                for i in range(0, robotsCount):
                    if not robotsDone[i]:
                        aTmp, heatmap = self.network.policy_action_certain(robotsOldState[i][0]) #i for selected robot, 0 beause the state is encapsulated once too much
                        a = np.ndarray.tolist(aTmp[0].numpy())
                        traingDataProximityCategories += [env.getMinDistOnVirtualRoadway(i)]
                        # traingDataProximityCategories += [env.getMinDistOnVirtualRoadwayWithScan(i, robotsOldState[i][0][3][0])]
                        # traingDataProximityCategories += [env.getRobotsProximityCategoryOnlyRobots(i)]
                        # traingDataProximityCategories += [env.getRobotsProximityCategoryAllObstacles(i)]
                        traingDataStates += [robotsOldState[i][0]]
                        if i == inspectedRobot:
                            proximityCategory = self.network.make_proximity_prediction(robotsOldState[i][0])[0].numpy()
                            #foo = env.getMinDistOnVirtualRoadway(0)
                            # foo = env.getMinDistOnVirtualRoadwayWithScan(0, np.asarray(robotsOldState[i][0][3][0])/env.simulation.robots[0].maxDistFact)
                            # proximityCategory = [0,0,0]
                            # proximityCategory[foo] = 1
                    else:
                        a = [None, None]
                    robotsActions.append(a)



                robotsStates = env.step(robotsActions, activations = None, proximity = proximityCategory)

                rewards = ''
                for i, stateData in enumerate(robotsStates):
                    new_state = stateData[0]
                    rewards += (str(i) + ': ' + str(stateData[1]) + '   ')
                    done = stateData[2]
                    robotsOldState[i] = new_state
                    if not robotsDone[i]:
                        robotsDone[i] = done
            if (e+1)%3 == 0:
                self.network.train_perception(traingDataStates, traingDataProximityCategories)
                traingDataStates = []
                traingDataProximityCategories = []

                #Saving enhanced perception model
                path = self.args.path
                path += 'PPO_enhanced_perception' + self.args.model_timestamp
                self.network.saveWeights(path)
                data = [self.args]
                with open(path + '.yml', 'w') as outfile:
                    yaml.dump(data, outfile, default_flow_style=False)