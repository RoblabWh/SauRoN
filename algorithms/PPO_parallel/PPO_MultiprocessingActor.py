import numpy as np
import yaml

from simulation.Environment import Environment
from algorithms.PPO_parallel.PPO_Network import PPO_Network
import os


class PPO_MultiprocessingActor:
    """
    The MultiprocessingActor is used during training to create and manage an own environment and simulation.
    Multiple Actors can be executed in parallel to create more training data for the neural net.
    To accomplish this the Multiprocessing Actor has an own copy of the trained neural net to calculate
    the actions for the robots in its simulation.
    After all observations are collected one (master) actor trains his network
    and the new weights are distributed to all remaining actors.
    """

    def __init__(self, app, act_dim, env_dim, args, weights, level, master):
        """
        Creates a multiprocessing actor
        :param act_dim: the number of continuous action dimensions (e.g. 2 for linear and angular velocity)
        :param env_dim: the number of input values for the neural net send by the environment
        :param args:
        :param weights: weights for the neural network. Only needed if the actor is not the master actor
        :param level: int - selected map level
        :param master: boolean - the master actor is used for training of the network weights and sets the initial weights
        """

        # Ray setzt die env Variable für die GPU selber (auf 0 bei einer GPU).
        # GPU kann nicht fehlerfrei genutzt werden und bietet teilweise keinen Leistungsvorteil während der Simulation
        # os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

        self.args = args
        self.app = app

        self.network = PPO_Network(act_dim, env_dim, args)
        self.network.build()
        #if master:
            #self.network.print_summary()

        if weights != None:
            self.network.set_model_weights(weights)

        self.env = Environment(self.app, args, env_dim[0], level)
        self.env.setUISaveListener(self)
        self.numbOfRobots = self.env.simulation.getCurrentNumberOfRobots()

        self.gamma = args.gamma
        self.reachedTargetList = [False] * 100
        self.level = level
        self.currentEpisode = -1
        self.cumul_reward = 0
        self.steps = 0
        self.resetActor()
        self.closed = False

    def setWeights(self, weights):
        self.network.set_model_weights(weights)

    def getWeights(self):
        return self.network.get_model_weights()

    def saveWeights(self, path):
        self.network.save_model_weights(path)

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

    def get_robots_state(self):
        # Robot 0 actions --> robotsData[0][0]
        # Robot 0 observations  --> robotsData[0][1]
        # Robot 0 rewards --> robotsData[0][2]
        # Robot 1 actions --> robotsData[1][0]
        # ...
        robotsData = []
        old_observations = []

        if self.reset:
            # Reset episode
            self.reset = False

            for i in range(self.numbOfRobots):
                old_observation = self.env.get_observation(i)
                old_observations.append(np.expand_dims(np.asarray(old_observation, dtype=object), axis=0))

                actions, states, rewards, env_done, evaluation, neglog = [], [], [], [], [], []
                robotsData.append([actions, states, rewards, env_done, evaluation, neglog])
        else:
            old_observations = self.robotsOldStateBackup

            # geht sicher schöner
            for robotDataBackup in self.robotsDataBackup:
                actions, states, rewards, env_done, evaluation, neglog = robotDataBackup
                robotsData.append([[actions[-1]],[states[-1]],[rewards[-1]], [env_done[-1]], [evaluation[-1]], [neglog[-1]]])

        return robotsData, old_observations

    def take_steps_in_env(self, numbrOfSteps):
        """
        Executes the simulation for a given number of steps
        :param numbrOfSteps: int - determines how many steps per robot are executed
        :return: collected experiences of all robots acting in this simulation
        """
        stepsLeft = numbrOfSteps
        cumul_reward = 0
        robotsData, old_observations = self.get_robots_state()

        while stepsLeft > 0 and not self.env.is_done():

            # Actor picks an action (following the policy)
            robotsActions = []  # actions of every Robot in the selected environment
            for i in range(0, len(robotsData)):  # iterating over every robot

                if not True in robotsData[i][3]:
                    aTmp = self.policy_action(old_observations[i][0])
                    action = np.ndarray.tolist(aTmp[0].numpy())[0]  # Tensoren in Numpy in List umwandeln
                    value = np.ndarray.tolist(aTmp[1].numpy())[0]
                    negL = np.ndarray.tolist(aTmp[2].numpy())

                else:
                    action = [None, None]
                robotsActions.append(action)

                if not None in action:
                    robotsData[i][0].append(action)
                    robotsData[i][4].append(value)
                    robotsData[i][5].append(negL)

            # environment makes a step with selected actions
            results = self.env.step(robotsActions)

            for i, dataCurrentFrameSingleRobot in enumerate(results):

                if not True in robotsData[i][3]:  # [environment] [robotsData (anstelle von OldState (1)] [Roboter] [done Liste]

                    new_observations = dataCurrentFrameSingleRobot[0]
                    reward = dataCurrentFrameSingleRobot[1]
                    env_done = dataCurrentFrameSingleRobot[2]
                    robotsData[i][1].append(old_observations[i][0])
                    robotsData[i][2].append(reward)
                    robotsData[i][3].append(env_done)
                    if env_done:
                        # just for tqdm's successrate...
                        reachedPickup = dataCurrentFrameSingleRobot[3]
                        self.reachedTargetList.pop(0)
                        self.reachedTargetList.append(reachedPickup)
                    # Update current state
                    old_observations[i] = new_observations
                    #print(robotsData[i][1][-1])
                    cumul_reward += reward
            stepsLeft -= 1
            self.steps += 1

        self.robotsDataBackup = robotsData
        self.robotsOldStateBackup = old_observations
        self.cumul_reward += cumul_reward


    def trainSteps(self, numbrOfSteps):
        """
        Executes the simulation for a given number of steps
        :param numbrOfSteps: int - determines how many steps per robot are executed
        :return: collected experiences of all robots acting in this simulation
        """
        stepsLeft = numbrOfSteps
        cumul_reward = 0
        robotsData = []
        old_observations = []

        if self.reset:
            # Reset episode
            self.reset = False

            for i in range(self.numbOfRobots):
                old_observation = self.env.get_observation(i)
                old_observations.append(np.expand_dims(np.asarray(old_observation, dtype=object), axis=0))

                actions, states, rewards, env_done, evaluation, neglog = [], [], [], [], [], []
                robotsData.append([actions, states, rewards, env_done, evaluation, neglog])
            # Robot 0 actions --> robotsData[0][0]
            # Robot 0 observations  --> robotsData[0][1]
            # Robot 0 rewards --> robotsData[0][2]
            # Robot 1 actions --> robotsData[1][0]
            # ...
        else:
            old_observations = self.robotsOldStateBackup

            # geht sicher schöner
            for robotDataBackup in self.robotsDataBackup:
                actions, states, rewards, env_done, evaluation, neglog = robotDataBackup
                robotsData.append([[actions[-1]],[states[-1]],[rewards[-1]], [env_done[-1]], [evaluation[-1]], [neglog[-1]]])

        while stepsLeft > 0 and not self.env.is_done():

            # Actor picks an action (following the policy)
            robotsActions = []  # actions of every Robot in the selected environment
            for i in range(0, len(robotsData)):  # iterating over every robot

                if not True in robotsData[i][3]:
                    aTmp = self.policy_action(old_observations[i][0])
                    action = np.ndarray.tolist(aTmp[0].numpy())[0]  # Tensoren in Numpy in List umwandeln
                    value = np.ndarray.tolist(aTmp[1].numpy())[0]
                    negL = np.ndarray.tolist(aTmp[2].numpy())

                else:
                    action = [None, None]
                robotsActions.append(action)

                if not None in action:
                    robotsData[i][0].append(action)
                    robotsData[i][4].append(value)
                    robotsData[i][5].append(negL)

            # environment makes a step with selected actions
            results = self.env.step(robotsActions)

            for i, dataCurrentFrameSingleRobot in enumerate(results):

                if not True in robotsData[i][3]:  # [environment] [robotsData (anstelle von OldState (1)] [Roboter] [done Liste]

                    new_observations = dataCurrentFrameSingleRobot[0]
                    reward = dataCurrentFrameSingleRobot[1]
                    env_done = dataCurrentFrameSingleRobot[2]
                    robotsData[i][1].append(old_observations[i][0])
                    robotsData[i][2].append(reward)
                    robotsData[i][3].append(env_done)
                    if env_done:
                        # just for tqdm's successrate...
                        reachedPickup = dataCurrentFrameSingleRobot[3]
                        self.reachedTargetList.pop(0)
                        self.reachedTargetList.append(reachedPickup)
                    # Update current state
                    old_observations[i] = new_observations
                    #print(robotsData[i][1][-1])
                    cumul_reward += reward
            stepsLeft -= 1
            self.steps += 1
        self.robotsDataBackup = robotsData
        self.robotsOldStateBackup = old_observations
        self.cumul_reward += cumul_reward

        return self.restructureRobotsData(robotsData)

    def restructureRobotsData(self, robotsData):
        """

        restructures the collected experiences of every robot in this simulation into a combined experience

        :param robotsData: list with experiences from every robot
        :return: python dictionary with the collected and restructured experience of this remote actors simulation
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
            usedTimeSteps = [] # currently not used!

            for s in states:
                #laser = np.array([np.array(s[i][0]) for i in range(0, len(s))]).swapaxes(0,1)
                laser = np.array([np.array(s[i][0]) for i in range(0, len(s))]).swapaxes(0, 2)
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

            # maybe ändern
            # advantage richtig ausgerechnet?
            advantagesTmp = discounted_rewardsTmp - np.reshape(evaluations, len(evaluations))
            advantagesTmp = (advantagesTmp - advantagesTmp.mean()) / (advantagesTmp.std() + 1e-10)
            advantages = np.concatenate((advantages, advantagesTmp))

        observation = {'lidar_0': statesConcatenatedL, 'orientation_to_goal': statesConcatenatedO, 'distance_to_goal': statesConcatenatedD, 'velocity': statesConcatenatedV}
        exp = {'observation': observation, 'action':actionsConcatenated, 'neglog_policy':neglogsConcatinated, 'reward':discounted_rewards, 'advantage':advantages}
        return exp

    def discount(self, rewards):
        """
        Compute the gamma-discounted rewards over an episode
        """
        t_steps = np.arange(len(rewards))
        r = rewards * self.gamma ** t_steps
        r = r[::-1].cumsum()[::-1] / self.gamma ** t_steps

        return r

    # def discount(self, r):
    #     """
    #     Compute the gamma-discounted rewards over an episode
    #     """
    #     discounted_r = np.zeros_like(r, dtype=float)
    #     cumul_r = 0
    #     for t in reversed(range(0, len(r))):
    #         cumul_r = r[t] + cumul_r * self.gamma
    #         discounted_r[t] = cumul_r
    #     return discounted_r

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

    def policy_action(self, s):
        """
        Use the actor to predict the next action to take, using the policy
        :param s: current state of a single robot
        :return: [actions, critic]
        """

        #laser = np.array([np.array(s[i][0]) for i in range(0, len(s))]).swapaxes(0,1)
        laser = np.array([np.array(s[i][0]) for i in range(0, len(s))]).swapaxes(0,2)
        #print("laser_state1: ", laser.shape)
        orientation = np.array([np.array(s[i][1]) for i in range(0, len(s))]).swapaxes(0,1)
        distance = np.array([np.array(s[i][2]) for i in range(0, len(s))])
        velocity = np.array([np.array(s[i][3]) for i in range(0, len(s))]).swapaxes(0,1)
        #print(np.array([laser]))
        #print(np.array([laser]).shape)
        return self.network.predict(np.array([laser]), np.array([orientation]), np.array([[distance]]), np.array([velocity]))  # Liste mit [actions, value]

    def train_net_obs(self, obs_with_actions_list):
        """
        Traines the network based on all collected experiences found inside of the observation list

        :param obs_with_actions_list: list with all observations used for training
        :return: new model weights
        """
        self.network.train(obs_with_actions_list['observation'], obs_with_actions_list)
        return self.network.get_model_weights()

    def killActor(self):
        self.env.simulation.simulationWindow.close()

    def showWindow(self):
        self.env.simulation.simulationWindow.show()
        self.env.simulation.hasUI = True

    def hideWindow(self):
        self.env.simulation.simulationWindow.hide()
        self.env.simulation.hasUI = False

    def isNotShowing(self):
        return self.app == None

    def has_been_closed(self):
        if self.closed:
            self.closed = False
            return True
        return False

    def window_closed(self):
        self.env.simulation.simulationWindow.close()
        self.closed = True

