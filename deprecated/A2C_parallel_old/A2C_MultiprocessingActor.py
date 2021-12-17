import ray
import numpy as np
import yaml

from simulation.Environment import Environment
from deprecated.A2C_parallel_old.A2C_Network import A2C_Network
import sys
from PyQt5.QtWidgets import QApplication
import os


@ray.remote
class A2C_MultiprocessingActor:
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
        app = None
        if master:
            app = QApplication(sys.argv)
        self.network = A2C_Network(act_dim, env_dim, args)
        if weights != None:
            self.network.setWeights(weights)
        self.env = Environment(app, args, env_dim[0]) #None --> No UI
        self.env.setUISaveListener(self)
        self.numbOfRobots = args.numb_of_robots
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
        self.network.setWeights(weights)

    def getWeights(self):
        self.network.getWeights()

    def saveWeights(self, path):
        self.network.saveWeights(path)

    def saveCurrentWeights(self):
        print('saving individual')
        path = self.args.path + 'A2C' + self.args.model_timestamp + "_e" + str(self.currentEpisode)
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

    def trainOneEpisode(self):
        # Reset episode
        zeit, cumul_reward, done = 0, 0, False

        self.env.reset(self.level)
        robotsData = []
        robotsOldState = []

        for i in range(self.numbOfRobots):
            old_state = self.env.get_observation(i)
            robotsOldState.append(np.expand_dims(old_state, axis=0))

            actions, states, rewards, done, evaluation = [], [], [], [], []
            robotsData.append((actions, states, rewards, done, evaluation))
        # Robot 0 actions --> robotsData[0][0]
        # Robot 0 states  --> robotsData[0][1]
        # Robot 0 rewards --> robotsData[0][2]
        # Robot 1 actions --> robotsData[1][0]
        # ...

        while not self.env.is_done():

            # Actor picks an action (following the policy)
            robotsActions = []  # actions of every Robot in the selected environment
            for i in range(0, len(robotsData)):  # iterating over every robot
                if not True in robotsData[i][3]:
                    aTmp = self.policy_action(robotsOldState[i][0])
                    a = np.ndarray.tolist(aTmp[0])[0]
                    c = np.ndarray.tolist(aTmp[1])[0]
                else:
                    a = [None, None]
                robotsActions.append(a)

                if not None in a:
                    robotsData[i][0].append(a)  # action_onehot) #TODO Tupel mit 2 werten von je -1 bis 1
                    robotsData[i][4].append(c)

            # environment makes a step with selected actions
            results = self.env.step(robotsActions)

            for i, dataCurrentFrameSingleRobot in enumerate(results):  # results[1] hat id, die hierfür nicht mehr gebraucht wird

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
            zeit += 1
        return robotsData

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

                actions, states, rewards, done, evaluation = [], [], [], [], []
                robotsData.append((actions, states, rewards, done, evaluation))
            # Robot 0 actions --> robotsData[0][0]
            # Robot 0 states  --> robotsData[0][1]
            # Robot 0 rewards --> robotsData[0][2]
            # Robot 1 actions --> robotsData[1][0]
            # ...
        else:

            robotsData = []
            robotsOldState = self.robotsOldStateBackup

            for robotDataBackup in self.robotsDataBackup:
                actions, states, rewards, done, evaluation = robotDataBackup
                robotsData.append(([actions[-1]],[states[-1]],[rewards[-1]], [done[-1]], [evaluation[-1]]))




        while stepsLeft > 0 and not self.env.is_done():

            # Actor picks an action (following the policy)
            robotsActions = []  # actions of every Robot in the selected environment
            for i in range(0, len(robotsData)):  # iterating over every robot
                if not True in robotsData[i][3]:
                    aTmp = self.policy_action(robotsOldState[i][0])
                    a = np.ndarray.tolist(aTmp[0])[0]
                    c = np.ndarray.tolist(aTmp[1])[0]
                else:
                    a = [None, None]
                robotsActions.append(a)

                if not None in a:
                    robotsData[i][0].append(a)
                    robotsData[i][4].append(c)

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
        return robotsData


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
            return self.network.predict(np.array([laser]), np.array([orientation]), np.array([distance]),
                                        np.array([velocity]))  # Liste mit [actions, value]


    def trainNet(self, statesConcatenatedL, statesConcatenatedO, statesConcatenatedD,statesConcatenatedV, statesConcatenatedT,discounted_rewards, actionsConcatenated,advantages):
        self.network.train_net(statesConcatenatedL, statesConcatenatedO, statesConcatenatedD,statesConcatenatedV, statesConcatenatedT,discounted_rewards, actionsConcatenated,advantages)
        return self.network.getWeights()

    ######################################################################################################
    #####Folgende Methoden werden bei der VErwendung eines zentralen Netzes im Hauptptozess benötigt #####
    ######################################################################################################
    #Das Netz aus dem Konstruktor wird dann nicht benötigt (kann auskommentiert werden)

    def doSingleStep(self, robotsActions, robotsCritics):

        robotsData = self.robotsData
        robotsOldState = self.robotsOldStateBackup

        for i, action in enumerate(robotsActions):
            if not None in action:
                robotsData[i][0].append(robotsActions[i])  # action
                robotsData[i][4].append(robotsCritics[i])  # critic value

        # environment makes a step with selected actions
        results = self.env.step(robotsActions)

        for i, dataCurrentFrameSingleRobot in enumerate(results):  # results[1] hat id, die hierfür nicht mehr gebraucht wird

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

        self.robotsOldStateBackup = robotsOldState
        self.robotsData = robotsData

        return (self.robotsData, robotsOldState)

    def getRobotDataAndReset(self):
        self.robotsDataBackup = self.robotsData.copy()

        robotsData = []
        for robotDataBackup in self.robotsDataBackup:
            actions, states, rewards, done, evaluation = robotDataBackup
            robotsData.append(([actions[-1]], [states[-1]], [rewards[-1]], [done[-1]], [evaluation[-1]]))
        self.robotsData = robotsData

        return self.robotsDataBackup

    def getInitialObservation(self):
        self.resetActor()
        robotsData = []
        robotsOldState = []

        for i in range(self.numbOfRobots):
            old_state = self.env.get_observation(i)
            robotsOldState.append(np.expand_dims(old_state, axis=0))

            actions, states, rewards, done, evaluation = [], [], [], [], []
            robotsData.append((actions, states, rewards, done, evaluation))

        self.robotsData = robotsData
        self.robotsOldStateBackup = robotsOldState
        return (robotsData, robotsOldState)

