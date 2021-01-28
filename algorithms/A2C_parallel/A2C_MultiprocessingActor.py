import ray
import numpy as np
from EnvironmentWithUI import Environment
from algorithms.A2C_parallel.A2C_Network import A2C_Network

@ray.remote
class A2C_MultiprocessingActor:

    def __init__(self, act_dim, env_dim, args, weights):
        self.args = args
        self.network = A2C_Network(act_dim, env_dim, args)
        self.network.setWeights(weights)
        self.env = Environment(None, args, env_dim[0], 0) #None --> No UI
        self.numbOfRobots = args.numb_of_robots
        self.timePenalty = args.time_penalty
        # self.av_meter = AverageMeter()
        self.gamma = args.gamma
        self.rechedTargetList = [False] * 100 #TODO mit erfolgsliste aus main Process zusammenlegen


    def setWeights(self, weights):
        self.network.setWeights(weights)


    def trainOneEpisode(self):
        # Reset episode
        zeit, cumul_reward, done = 0, 0, False

        self.env.reset()
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
                    aTmp = self.policy_action(robotsOldState[i][0], (self.rechedTargetList).count(True) / 100)
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

            for i, dataCurrentFrameSingleRobot in enumerate(results[0]):  # results[1] hat id, die hierfür nicht mehr gebraucht wird

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
                        self.rechedTargetList.pop(0)
                        self.rechedTargetList.append(reachedPickup)
                    # Update current state
                    robotsOldState[i] = new_state
                    cumul_reward += r
            zeit += 1
        return robotsData


    def policy_action(self, s, successrate):  # TODO obs_timestep mit übergeben
        """ Use the actor to predict the next action to take, using the policy
        """
        # std = ((1-successrate)**2)*0.55

        laser = np.array([np.array(s[i][0]) for i in range(0, len(s))])
        orientation = np.array([np.array(s[i][1]) for i in range(0, len(s))])
        distance = np.array([np.array(s[i][2]) for i in range(0, len(s))])
        velocity = np.array([np.array(s[i][3]) for i in range(0, len(s))])
        timesteps = np.array([np.array(s[i][4]) for i in range(0, len(s))])
        # print(laser.shape, orientation.shape, distance.shape, velocity.shape)
        if (self.timePenalty):
            # Hier breaken um zu gucken, ob auch wirklich 4 timeframes hier eingegeben werden oder was genau das kommt
            return self.network.predict(np.array([laser]), np.array([orientation]), np.array([distance]),
                                        np.array([velocity]), np.array([timesteps]))  # Liste mit [actions, value]
        else:
            return self.network.predict(np.array([laser]), np.array([orientation]), np.array([distance]),
                                        np.array([velocity]))  # Liste mit [actions, value]

