import numpy as np
from PyQt5.QtWidgets import QApplication
from simulation.Environment import Environment
from deprecated.A2C_parallel_old.A2C_Multi import AverageMeter
from algorithms.PPO_parallel.PPO_MultiprocessingActor import PPO_MultiprocessingActor
from algorithms.PPO_parallel.PPO_Network import PPO_Network
from tqdm import tqdm
import yaml
import tensorflow
from random import shuffle
import multiprocessing


import sys
# insert at 1, 0 is the script path (or '' in REPL)

sys.path.insert(1, '/')


class PPO_Multi:
    """
    Defines an Proximal Policy Optimization learning algorithm for neural nets
    """

    def __init__(self, app, act_dim, env_dim, args):
        """
        :param act_dim: number of available actions
        :param env_dim: (number of past states (including the current one), size of a state) -
            their product determines the number input neurons
        :param args:
        """
        self.args = args
        self.app = app
        self.levelFiles = args.level_files
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.numbOfParallelEnvs = args.parallel_envs
        self.numbOfRobots = args.numb_of_robots
        self.av_meter = AverageMeter()
        self.gamma = args.gamma
        self.closed_windows = []
        self.successrate = 0


    def prepare_training(self, loadWeightsPath = ""):
        """
        Will be called before training.
        creates remote workers for running parallel simulations
        :param loadWeightsPath: path to pretrained weights. If left empty new random start weights will be created
        :return: Tupel (False  for not finished training yet, List with names of loaded levels per environment)
        """
        self.currentEpisode = 0

        loadedWeights = None
        if loadWeightsPath != "":
            if self.args.load_weights_only:
                loadWeightsPath += '.h5'

            self.load_net(loadWeightsPath)
            loadedWeights = self.network.get_model_weights()
            tensorflow.keras.backend.clear_session()

        #Create parallel workers with own environment
        envLevel = [int(i/(self.numbOfParallelEnvs/len(self.levelFiles))) for i in range(self.numbOfParallelEnvs)]


        multiActors = [PPO_MultiprocessingActor(self.app, self.act_dim, self.env_dim, self.args, loadedWeights, envLevel[0], True)]
        #startweights = multiActors[0].getWeights()
        #multiActors += [PPO_MultiprocessingActor.remote(self.act_dim, self.env_dim, self.args, startweights, envLevel[i + 1], False) for i in range(self.numbOfParallelEnvs - 1)]
       
        levelNames = []
        for i, actor in enumerate(multiActors):
            levelName = actor.setLevel(envLevel[i])
            levelNames.append(levelName)

        self.multiActors = multiActors
        self.activeActors = multiActors


        # Main Loop
        self.tqdm_e = tqdm(range(self.args.nb_episodes), desc='Score', leave=True, unit=" episodes")

        return (False, levelNames)


    def train_with_feedback_for_n_steps(self, visibleLevels):
        """
        Runs the simulation in every remote worker for n steps (defined in args)
        and uses collected Data for training the Network afterwards
        :param visibleLevels: list with true or false that determines which simulations have to be visualized
        :return: boolean that stands for beeing still active or done with whole episode
        """

        activeActors = self.activeActors
        if len(activeActors) > 0:
            allTrainingResults = [actor.trainSteps(self.args.train_interval) for actor in activeActors] #Liste mit Listen von Observations aus den Environments
            trainedWeights = self.train_models_with_obs(allTrainingResults, self.multiActors[0])

            self.multiActors[0].setWeights(trainedWeights)
            #for actor in self.multiActors[1:len(self.multiActors)]:
            #    actor.setWeights(trainedWeights)

            activeActors = []
            for actor in self.multiActors:
                if actor.isActive():
                    activeActors.append(actor)


            for i, show in enumerate(visibleLevels):
                if show:
                    if self.multiActors[i].has_been_closed():
                        self.closed_windows.append(i)
                    else:
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

        self.tqdm_e.update(1)
        self.currentEpisode += 1
        #reset the active actors list for next episode
        self.activeActors = self.multiActors

        #save weights in predefined interval
        if (self.currentEpisode) % self.args.save_intervall == 0:
            print('Saving')
            self.save_weights(self.multiActors[0], self.args.path)

        #build statistics of last episode
        zeit, cumul_reward, done = 0, 0, False

        allReachedTargetList = []
        individualSuccessrate = []
        for actor in self.multiActors:
            tmpTargetList = actor.getTargetList()
            allReachedTargetList += tmpTargetList
            individualSuccessrate.append(tmpTargetList.count(True) / 100)

        targetDivider = (self.numbOfParallelEnvs) * 100  # Erfolg der letzten 100
        self.successrate = allReachedTargetList.count(True) / targetDivider

        # Calculate and display score
        individualLastAverageReward = []
        print("Update progressbar")
        for actor in self.multiActors:
            (cumRewardActor, steps) = actor.resetActor()
            self.av_meter.update(cumRewardActor, steps)
            cumul_reward += cumRewardActor
            individualLastAverageReward.append(cumRewardActor/steps)

        cumul_reward = cumul_reward / self.args.parallel_envs

        self.tqdm_e.set_description("R avr last e: " + str(cumul_reward) + " --R avr all e : " + str(self.av_meter.avg) + " --Avr Reached Target (25 epi): " + str(self.successrate))
        self.tqdm_e.refresh()

        done = False
        if self.currentEpisode == self.args.nb_episodes:
            # If all episodes are finished the current weights are saved
            self.save_weights(self.multiActors[0], self.args.path)

            for actor in self.multiActors:
                actor.killActor()

            done = True

        return done, individualLastAverageReward, individualSuccessrate, self.currentEpisode, self.successrate

    def get_closed_windows(self):
        tmp_list = self.closed_windows.copy()
        self.closed_windows = []
        return tmp_list

    def train_models_with_obs(self, obs_lists, master_env):
        """
        restructures the collected Data from every remote actor into one batch

        :param obs_lists: list with collected exeriences of every remote actor
        :param master_env: refernce to the master environment whose network will be used for training
        :return: trained weights
        """
        shuffle(obs_lists)
        numb_of_exp_per_batch = len(obs_lists)# int(1024 / (self.args.train_interval  * 4))

        obs_concatinated = []
        current_index = -1

        weights = None
        for i, exp in enumerate(obs_lists):
            if(i%numb_of_exp_per_batch == 0):
                current_index += 1
                obs_concatinated.append(exp)
            else:
                for key, value in obs_concatinated[current_index].items():
                    if type(value) == type(exp):
                        for key2, value2 in value.items():
                            obs_concatinated[current_index][key][key2] = np.concatenate((value2, exp[key][key2]))
                    else:
                        obs_concatinated[current_index][key] = np.concatenate((value, exp[key]))

        for exp in obs_concatinated:
            weights = master_env.train_net_obs(exp)
        return weights

    def saveCurrentWeights(self):
        """
        saves weights at current epoch
        """
        print('Saving individual')
        self.save_weights(self.args.path, "_e" + str(self.currentEpisode))

    def save_weights(self, masterEnv, path, additional=""):
        path += 'PPO' + self.args.model_timestamp + additional

        masterEnv.saveWeights(path)
        data = [self.args]
        with open(path+'.yml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

    def load_net(self, path):
        self.network = PPO_Network(self.act_dim, self.env_dim, self.args)
        self.network.build()

        if 'h5' in path[-2:]:
            self.network.load_weights(path)
        else:
            self.network.load_model(path)
        return True

    def showEnvWindow(self, envID):
        if envID < len(self.multiActors):
            self.multiActors[envID].showWindow()

    def hideEnvWindow(self, envID):
        if envID < len(self.multiActors):
            self.multiActors[envID].hideWindow()

    def execute(self, args, env_dim):
        """
        executes a trained net (without using the standard deviation in the action selection)

        :param args: args defined in main
        :param env_dim:
        """
        app = QApplication(sys.argv)
        env = Environment(app, args, env_dim, 0)
        env.simulation.showWindow(app)
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
                        aTmp, heatmap = self.network.pedict_certain(robotsOldState[i][0])
                        a = np.ndarray.tolist(aTmp[0].numpy())
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
        Trains a pretrained network to detect other robots or obstacles in a straight corridor in front of the robot.
        During the training only the convolutional layers are optimized

        :param args: args defined in main
        :param env_dim:
        """
        app = QApplication(sys.argv)
        env = Environment(app, args, env_dim, 0)
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
                        aTmp, heatmap = self.network.pedict_certain(robotsOldState[i][0])
                        a = np.ndarray.tolist(aTmp[0].numpy())
                        traingDataProximityCategories += [env.getMinDistOnVirtualRoadway(i)]

                        traingDataStates += [robotsOldState[i][0]]
                        if i == inspectedRobot:
                            proximityCategory = self.network.make_proximity_prediction(robotsOldState[i][0])[0].numpy()

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
                self.network.save_model_weights(path)
                data = [self.args]
                with open(path + '.yml', 'w') as outfile:
                    print("save model at episode ", e)
                    yaml.dump(data, outfile, default_flow_style=False)