import sys

import numpy as np
from PyQt5.QtWidgets import QApplication

from simulation import Environment
from BucketRenderer import BucketRenderer
#from algorithms.PPO_parallel.A2C_Network import A2C_Network
#from algorithms.PPO_parallel.A2C_MultiprocessingActor import A2C_MultiprocessingActor
from tqdm import tqdm
import ray
import yaml
#import keras



class A2C_Multi:
    """
    Defines an Advantage Actor Critic learning algorithm for neural nets
    """

    def __init__(self, act_dim, env_dim, args):
        """
        :param act_dim: number of available actions
        :param env_dim: (number of past states (including the current one), size of a state) -
            their product determines the number input neurons
        :param args:
        """
        self.args = args
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.numbOfParallelEnvs = args.parallel_envs
        self.numbOfRobots = args.numb_of_robots
        self.timePenalty = args.time_penalty
        self.av_meter = AverageMeter()
        self.gamma = args.gamma

    def train(self, loadWeightsPath=""):
        """
        Main A2C Training Algorithm

        This implementation of an A2C uses the MultiprocessingActor class to parallelize the simulations.
        Therefore the "Ray" API is used. In the beginning of the training process a number of ray workers (in form of the
        MultiprocessingActors are created) and receive their initial weights and a level that determines the level layout.

        In each episode the actors collect experiences for all of their robots for n steps (defined in the main).
        These experiences are then used to train the network of one MultiprocessingActor (the master-actor).
        After that the new weights get distributed to the other actors in order to update their weights for the next training.
        This process will repeat as long as there are active MultiprocessingActors. An actor object is active until its
        environment becomes inactive (which happens after all robots crashed/ reached their goal or the number of total steps
        per episode is consumed).



        :param loadWeightsPath: The path to the .h5 file containing the pretrained weights.
         Only required if a pretrained net is used.
        """

        loadedWeights = None
        if loadWeightsPath != "":
            self.load_net(loadWeightsPath)
            loadedWeights = self.network.getWeights()
            keras.backend.clear_session()


        #Create parallel workers with own environment
        envLevel = [i%9 for i in range(self.numbOfParallelEnvs)]
        ray.init()
        multiActors = [A2C_MultiprocessingActor.remote(self.act_dim, self.env_dim, self.args, None, envLevel[0], True)]
        startweights = multiActors[0].getWeights.remote()
        multiActors += [A2C_MultiprocessingActor.remote(self.act_dim, self.env_dim, self.args, startweights, envLevel[i+1], False) for i in range(self.numbOfParallelEnvs-1)]
        for i, actor in enumerate(multiActors):
            actor.setLevel.remote(envLevel[i])

        # Main Loop
        tqdm_e = tqdm(range(self.args.nb_episodes), desc='Score', leave=True, unit=" episodes")

        for e in tqdm_e:
            self.currentEpisode = e
            #Start of episode for the parallel A2C actors with their own environment
            # HIer wird die gesamte Episode durchlaufen und dann erst trainiert

            #Training bereits alle n=75 steps mit den zu dem Zeitounkt gesammelten Daten (nur für noch aktive Environments)
            activeActors = multiActors
            while len(activeActors) > 0:
                futures = [actor.trainSteps.remote(self.args.train_interval) for actor in activeActors]

                allTrainingResults = ray.get(futures)
                trainedWeights = self.train_models(allTrainingResults, multiActors[0])
                for actor in multiActors[1:len(multiActors)]:
                    actor.setWeights.remote(trainedWeights)

                activeActors = []
                for actor in multiActors:
                    if ray.get(actor.isActive.remote()):
                        activeActors.append(actor)


            # Reset episode
            zeit, cumul_reward, done = 0, 0, False

            if (e+1) % self.args.save_intervall == 0:
                print('Saving')
                self.save_weights(multiActors[0], self.args.path)

            allReachedTargetList = []

            for actor in multiActors:
                tmpTargetList = ray.get(actor.getTargetList.remote())
                allReachedTargetList += tmpTargetList

            # allTrainingResults.append(robotsData)
            targetDivider = (self.numbOfParallelEnvs) * 100  # Erfolg der letzten 100
            successrate = allReachedTargetList.count(True) / targetDivider


            for actor in multiActors:
                (cumRewardActor, steps) = ray.get(actor.resetActor.remote())
                self.av_meter.update(cumRewardActor, steps)
                cumul_reward += cumRewardActor
            cumul_reward = cumul_reward / self.args.parallel_envs


            # Display score
            tqdm_e.set_description("R avr last e: " + str(cumul_reward) + " -- R avr all e : " + str(self.av_meter.avg) + " Avr Reached Target (25 epi): " + str(successrate))
            tqdm_e.refresh()


    def train_models(self, envsData, masterEnv = None):#, states, actions, rewards): 1 0 2
        """
        Update actor and critic networks from experience
        :param envsData: Collected states of all robots from all used parallel environments. Collected over the last n time steps
        :param masterEnv: The environment which is used to train the network. all other networks will receive a copy
        of its weights after the training process.
        :return: trained weights of the master environments network
        """

        discounted_rewards = np.array([])
        state_values = np.array([])
        advantages = np.array([])
        actionsConcatenated = np.array([])
        statesConcatenatedL = np.array([])
        statesConcatenatedO = np.array([])
        statesConcatenatedD = np.array([])
        statesConcatenatedV = np.array([])
        statesConcatenatedT = np.array([])
        for robotsData in envsData:
            for data in robotsData:
                actions, states, rewards, dones, evaluations = data

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
                else:
                    statesConcatenatedL = np.concatenate((statesConcatenatedL, np.array(lasers)))
                    statesConcatenatedO = np.concatenate((statesConcatenatedO, np.array(orientations)))
                    statesConcatenatedD = np.concatenate((statesConcatenatedD, np.array(distances)))
                    statesConcatenatedV = np.concatenate((statesConcatenatedV, np.array(velocities)))
                    statesConcatenatedT = np.concatenate((statesConcatenatedT, np.array(usedTimeSteps)))
                    state_values = np.concatenate((state_values, evaluations))

                discounted_rewardsTmp = self.discount(rewards)
                discounted_rewards = np.concatenate((discounted_rewards, discounted_rewardsTmp))



                advantagesTmp = discounted_rewardsTmp - np.reshape(evaluations, len(evaluations))  # Warum reshape
                advantagesTmp = (advantagesTmp - advantagesTmp.mean()) / (advantagesTmp.std() + 1e-8)
                advantages = np.concatenate((advantages, advantagesTmp))


        if masterEnv == None:
            self.network.train_net(statesConcatenatedL, statesConcatenatedO, statesConcatenatedD,statesConcatenatedV, statesConcatenatedT,discounted_rewards, actionsConcatenated,advantages)
            weights = self.network.getWeights()
        else:
            weights = ray.get(masterEnv.trainNet.remote(statesConcatenatedL, statesConcatenatedO, statesConcatenatedD,statesConcatenatedV, statesConcatenatedT,discounted_rewards, actionsConcatenated,advantages))
        return weights

    def discount(self, r):
        """
        Compute the gamma-discounted rewards over a training batch
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
        path += 'A2C' + self.args.model_timestamp + additional

        masterEnv.saveWeights.remote(path)
        data = [self.args]
        with open(path+'.yml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

    def load_net(self, path):
        self.network = A2C_Network(self.act_dim, self.env_dim, self.args)
        self.network.load_weights(path)

        return True

    def execute(self,  args, env_dim):
        """
        executes a trained net (without using the standard deviation in the action selection)
        :param env: EnvironmentWithUI.Environment
        :param args: args defined in main
        """
        # visualization of chosen actions

        app = QApplication(sys.argv)
        env = Environment.Environment(app, args, env_dim, 0)
        env.simulation.showWindow()

        histogramm = BucketRenderer(20, 0)
        histogramm.show()
        liveHistogramRobot = 0

        robotsCount = self.numbOfRobots


        for e in range(18):

            env.reset(0)
            robotsOldState = [np.expand_dims(env.get_observation(i), axis=0) for i in range(0, robotsCount)]
            robotsDone = [False for i in range(0, robotsCount)]

            while not env.is_done():

                robotsActions = []
                # Actor picks an action (following the policy)
                for i in range(0, robotsCount):

                    if not robotsDone[i]:
                        aTmp = self.network.policy_action_certain(robotsOldState[i][0])
                        a = np.ndarray.tolist(aTmp[0])[0]

                        # visualization of chosen actions
                        if i == liveHistogramRobot:
                            histogramm.add_action(a)
                            histogramm.show()

                    else:
                        a = [None, None]

                    robotsActions.append(a)

                robotsStates= env.step(robotsActions)


                rewards = ''
                for i, stateData in enumerate(robotsStates):
                    new_state = stateData[0]
                    rewards += (str(i) + ': ' + str(stateData[1]) + '   ')
                    done = stateData[2]

                    robotsOldState[i] = new_state
                    if not robotsDone[i]:
                        robotsDone[i] = done
        return False



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

