import ray
import numpy as np
import yaml

from EnvironmentWithUI import Environment
#from algorithms.A2C_parallel.PPO_Network import PPO_Network
from algorithms.A2C_parallel.PPO_Network_NewContinuousLayer import PPO_Network
from algorithms.A2C_parallel.robins.A2C_Network_robin import Robin_Network
import sys
from PyQt5.QtWidgets import QApplication
import os
import tensorflow as tf
import time


@ray.remote
class PPO_MultiprocessingActor:
    """
    The MultiprocessingActor is used during training to create and manage an own environment and simulation.
    Multiple Actors can be executed in parallel to create more training data for the neural net.
    To accomplish this the Multiprocessing Actor has an own copy of the trained neural net to calculate
    the actions for the robots in its simulation.
    After all observations are collected one (master) actor trains his network
    and the new weights are distributed to all remaining actors.
    """

    def __init__(self, act_dim, env_dim, args, weights, level, master):
        """
        Creates a multiprocessing actor
        :param act_dim: the number of continuous action dimensions (e.g. 2 for linear and angular velocity)
        :param env_dim: the number of input values for the neural net send by the environment
        :param args:
        :param weights: weights for the neural network. Only needed if the actor is not the master actor
        :param level: int - selected map level
        :param master: boolean - the master actor is used for training of the network weights and sets the initial weights
        """

        # Tensorflow-GPU: 2.2.0 muss installiert sein
        # Dafür wird Cuda 10.1 benötigt
        #     -> Hinweis dafür muss zusätzlich cuDNN für Cuda 10.1 installiert werden
        #        (die dort enthaltenen Dateien in das Nvidia Toolkit über den Filebrowser einfügen)
        #        https://medium.com/@sunnydhoke22/how-to-install-cuda-10-and-cudnn-for-tensorflow-gpu-on-windows-10-414c10eabc96

        # Ray setzt die env Variable für die GPU selber (auf 0 bei einer GPU).
        # Soll sie nicht verwendet werden muss sie manuell auf -1 gesetzt werden:
        if not args.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


        self.args = args
        self.app = None
        #self.network = PPO_Network(act_dim, env_dim, args)
        self.network = Robin_Network(act_dim, env_dim, args) #Robin
        self.network.build()
        if master:
            self.app = QApplication(sys.argv)
            # self.network.printSummary()
            self.network.print_summary() #Robin
        if weights != None:
            # self.network.setWeights(weights)
            self.network.set_model_weights(weights) #Robin
        self.env = Environment(self.app, args, env_dim[0], level)
        self.env.setUISaveListener(self)
        self.numbOfRobots = self.env.simulation.getCurrentNumberOfRobots()
        self.timePenalty = args.time_penalty
        # self.av_meter = AverageMeter()
        self.gamma = args.gamma
        self.reachedTargetList = [False] * 100
        self.level = level
        self.currentEpisode = -1
        self.cumul_reward = 0
        self.steps = 0
        self.resetActor()


    def setWeights(self, weights):
        # self.network.setWeights(weights)
        self.network.set_model_weights(weights) #Robins

    def getWeights(self):
        # return self.network.getWeights()
        return self.network.get_model_weights() #Robins

    def saveWeights(self, path):
        # self.network.saveWeights(path)
        self.network.save_model_weights(path) # Robin

    def saveCurrentWeights(self):
        print('saving individual')
        path = self.args.path + 'PPO' + self.args.model_timestamp + "_e" + str(self.currentEpisode)
        self.saveWeights(path)

        data = [self.args]
        with open(path+'.yml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

    def getTargetList(self):
        return self.reachedTargetList

    def setLevel(self, level):
        self.level = level
        for _ in range(len(self.reachedTargetList)):
            self.reachedTargetList.pop(0)
            self.reachedTargetList.append(False)
        self.env.reset(level)

        return self.env.simulation.getLevelName()

    def trainSteps(self, numbrOfSteps):
        stepsLeft , cumul_reward = numbrOfSteps, 0

        if self.reset:
            self.reset = False
            # Reset episode
            done = False

            robotsData = []
            robotsOldState = []

            for i in range(self.numbOfRobots):
                old_state = self.env.get_observation(i)
                robotsOldState.append(np.expand_dims(old_state, axis=0))

                actions, states, rewards, done, evaluation, neglog = [], [], [], [], [], []
                robotsData.append((actions, states, rewards, done, evaluation, neglog))
            # Robot 0 actions --> robotsData[0][0]
            # Robot 0 states  --> robotsData[0][1]
            # Robot 0 rewards --> robotsData[0][2]
            # Robot 1 actions --> robotsData[1][0]
            # ...
        else:

            robotsData = []
            robotsOldState = self.robotsOldStateBackup

            for robotDataBackup in self.robotsDataBackup:
                actions, states, rewards, done, evaluation, neglog = robotDataBackup
                robotsData.append(([actions[-1]],[states[-1]],[rewards[-1]], [done[-1]], [evaluation[-1]], [neglog[-1]]))


        while stepsLeft > 0 and not self.env.is_done():

            # Actor picks an action (following the policy)
            robotsActions = []  # actions of every Robot in the selected environment
            for i in range(0, len(robotsData)):  # iterating over every robot
                if not True in robotsData[i][3]:
                    aTmp = self.policy_action(robotsOldState[i][0])
                    a = np.ndarray.tolist(aTmp[0].numpy())[0]  # Tensoren in Numpy in List umwandeln
                    #a = np.ndarray.tolist(aTmp[0])[0]
                    c = np.ndarray.tolist(aTmp[1].numpy())[0]
                    negL = np.ndarray.tolist(aTmp[2].numpy())
                    # print(aTmp, a, c, neglog)
                else:
                    a = [None, None]
                robotsActions.append(a)

                if not None in a:
                    robotsData[i][0].append(a)
                    robotsData[i][4].append(c)
                    robotsData[i][5].append(negL)

            # environment makes a step with selected actions
            results = self.env.step(robotsActions)

            for i, dataCurrentFrameSingleRobot in enumerate(
                    results):  # results[1] hat id, die hierfür nicht mehr gebraucht wird

                if not True in robotsData[i][3]:  # [environment] [robotsData (anstelle von OldState (1)] [Roboter] [done Liste]
                    # print("dataCurent Frame 0 of env",results[j][1], dataCurrentFrame[0])
                    new_state = dataCurrentFrameSingleRobot[0]
                    r = dataCurrentFrameSingleRobot[1]
                    done = dataCurrentFrameSingleRobot[2]
                    robotsData[i][1].append(robotsOldState[i][0])
                    robotsData[i][2].append(r)
                    robotsData[i][3].append(done)
                    if (done):
                        reachedPickup = dataCurrentFrameSingleRobot[3]
                        self.reachedTargetList.pop(0)
                        self.reachedTargetList.append(reachedPickup)
                    # Update current state
                    robotsOldState[i] = new_state
                    cumul_reward += r
            stepsLeft -= 1
            self.steps += 1
        self.robotsDataBackup = robotsData
        self.robotsOldStateBackup = robotsOldState
        self.cumul_reward += cumul_reward
        #return robotsData
        return self.restructureRobotsData(robotsData)
        #return self.reformat_observation(robotsData)


    def reformat_observation(self, robotsData):
        # start = time.time()

        all_obs_and_actions = []
        for data in robotsData: #data of a single roboter
            actions, states, rewards, dones, evaluations, neglogs = data
            discounted_rewards = self.discount(rewards)

            advantages = discounted_rewards - np.reshape(evaluations, len(evaluations))
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for j,s in enumerate(states):
                laser = np.array([np.array(s[i][0]) for i in range(0, len(s))]).swapaxes(0, 1)
                orientation = np.array([np.array(s[i][1]) for i in range(0, len(s))]).swapaxes(0, 1)
                distance = np.array([np.array(s[i][2]) for i in range(0, len(s))]).swapaxes(0, 1)
                velocity = np.array([np.array(s[i][3]) for i in range(0, len(s))]).swapaxes(0, 1)

                observation = {'laser_0': np.expand_dims(np.array(laser), 0),
                               'orientation_to_goal': np.expand_dims(np.array(orientation), 0),
                               'distance_to_goal': np.expand_dims(np.array(distance), 0),
                               'velocity': np.expand_dims(np.array(velocity), 0)}
                action = {'action': actions[j], 'value': evaluations[j], 'neglog_policy': neglogs[j],
                          'reward': discounted_rewards[j], 'advantage': advantages[j]}
                all_obs_and_actions += [(observation, action)]

        # end = time.time()
        # print('time for reformatting', self.level, end - start)

        return all_obs_and_actions


    def restructureRobotsData(self, robotsData):#, states, actions, rewards): 1 0 2
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

        # neglogsConcatinated = np.squeeze(neglogsConcatinated)


        #statesConcatenatedL, statesConcatenatedO, statesConcatenatedD,statesConcatenatedV, statesConcatenatedT,discounted_rewards, actionsConcatenated,advantages, neglogs)
        observation = {'lidar_0': statesConcatenatedL, 'orientation_to_goal': statesConcatenatedO, 'distance_to_goal': statesConcatenatedD, 'velocity': statesConcatenatedV} #alternativ viel einzelne observations?
        exp = {'observation': observation, 'action':actionsConcatenated, 'neglog_policy':neglogsConcatinated, 'reward':discounted_rewards, 'advantage':advantages}
        return exp#(statesConcatenatedL, statesConcatenatedO, statesConcatenatedD,statesConcatenatedV, statesConcatenatedT, discounted_rewards, actionsConcatenated, advantages, neglogsConcatinated, state_values)



        # if masterEnv == None:
        #     #for i in len(statesConcatenatedL):
        #      #   self.network.train_net(statesConcatenatedL[i], statesConcatenatedO[i], statesConcatenatedD[i],statesConcatenatedV[i], statesConcatenatedT[i],discounted_rewards[i], actionsConcatenated[i],advantages[i], neglogsConcatinated[i])
        #     self.network.train_net(statesConcatenatedL, statesConcatenatedO, statesConcatenatedD,statesConcatenatedV, statesConcatenatedT,discounted_rewards, actionsConcatenated,advantages, neglogsConcatinated)
        #     weights = self.network.getWeights()
        # else:
        #     weights, var = ray.get(masterEnv.trainNet.remote(statesConcatenatedL, statesConcatenatedO, statesConcatenatedD,statesConcatenatedV, statesConcatenatedT,discounted_rewards, actionsConcatenated,advantages, neglogsConcatinated))
        # return weights , var

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


    def resetActor(self):
        self.env.reset(self.level)
        self.reset = True
        self.currentEpisode +=1
        returnCumulReward, returnSteps = self.cumul_reward/4, self.steps
        self.cumul_reward = 0
        self.steps = 0

        return (returnCumulReward, returnSteps)

    def isActive(self):
        return not self.env.is_done()

    def policy_action(self, s):  # TODO obs_timestep mit übergeben
        """
        Use the actor to predict the next action to take, using the policy
        :param s: current state of a single robot
        :return: [actions, critic]
        """

        laser = np.array([np.array(s[i][0]) for i in range(0, len(s))]).swapaxes(0,1)
        orientation = np.array([np.array(s[i][1]) for i in range(0, len(s))]).swapaxes(0,1)
        distance = np.array([np.array(s[i][2]) for i in range(0, len(s))]).swapaxes(0,1)
        velocity = np.array([np.array(s[i][3]) for i in range(0, len(s))]).swapaxes(0,1)
        timesteps = np.array([np.array(s[i][4]) for i in range(0, len(s))])
        # print(laser.shape, orientation.shape, distance.shape, velocity.shape)
        if (self.timePenalty):
            # Hier breaken um zu gucken, ob auch wirklich 4 timeframes hier eingegeben werden oder was genau das kommt
            return self.network.predict(np.array([laser]), np.array([orientation]), np.array([distance]),
                                        np.array([velocity]), np.array([timesteps]))  # Liste mit [actions, value]
        else:
            return self.network.predict(np.array([laser]), np.array([orientation]), np.array([distance]), np.array([velocity]))  # Liste mit [actions, value]


    def trainNet(self, statesConcatenatedL, statesConcatenatedO, statesConcatenatedD,statesConcatenatedV, statesConcatenatedT,discounted_rewards, actionsConcatenated,advantages, neglogs, values):
        # self.network.train_net(statesConcatenatedL, statesConcatenatedO, statesConcatenatedD,statesConcatenatedV, statesConcatenatedT,discounted_rewards, actionsConcatenated,advantages, neglogs, values)
        #Robins:
        for i in range(len(statesConcatenatedL)):
            observation = {'laser_0': np.expand_dims(np.array(statesConcatenatedL[i]),0),
                           'orientation_to_goal': np.expand_dims(np.array(statesConcatenatedO[i]),0),
                           'distance_to_goal': np.expand_dims(np.array(statesConcatenatedD[i]),0),
                           'velocity': np.expand_dims(np.array(statesConcatenatedV[i]),0)}
            action = {'action': actionsConcatenated[i], 'value': values[i], 'neglog_policy': neglogs[i],
                      'reward':  discounted_rewards[i], 'advantage': advantages[i]}
            self.network.train(observation, action)

        # return self.network.getWeights()
        return self.network.get_model_weights() #Robin



    def train_net_obs(self, obs_with_actions_list):
        # print(len(obs_with_actions_list))
        # for obs_with_actions in obs_with_actions_list:
        #     obs, action = obs_with_actions
        #     self.network.train(obs, action)
        self.network.train(obs_with_actions_list['observation'], obs_with_actions_list)
        return self.network.get_model_weights()

    def killActor(self):
        ray.actor.exit_actor()

    def showWindow(self):
        if self.app == None:
            self.app = QApplication(sys.argv)
            self.env.simulation.showWindow(self.app)

    def hideWindow(self):
        if self.app != None:
            self.app = None
            self.env.simulation.closeWindow()

    def isNotShowing(self):
        return self.app == None