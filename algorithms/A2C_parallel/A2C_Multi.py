import numpy as np
import keras as k

from algorithms.A2C_parallel.A2C_Network import A2C_Network
from algorithms.A2C_parallel.A2C_MultiprocessingActor import A2C_MultiprocessingActor
from tqdm import tqdm
import ray
import yaml



class A2C_Multi:

    def __init__(self, act_dim, env_dim, args):
        self.args = args
        self.network = A2C_Network(act_dim, env_dim, args)
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.numbOfParallelEnvs = args.parallel_envs
        self.numbOfRobots = args.numb_of_robots
        self.timePenalty = args.time_penalty
        self.av_meter = AverageMeter()
        self.gamma = args.gamma

    def train(self, env):
        """ Main A2C Training Algorithm
        """
        reachedTargetList = [False] * 100
        # countEnvs = len(envs)

        ray.init()
        multiActors = [A2C_MultiprocessingActor.remote(self.act_dim, self.env_dim, self.args, self.network.getWeights()) for _ in range(self.numbOfParallelEnvs)]
        envLevel = [0 for _ in range(self.numbOfParallelEnvs+1)]
        env.setUISaveListener(self)

        # Main Loop
        tqdm_e = tqdm(range(self.args.nb_episodes), desc='Score', leave=True, unit=" episodes")

        for e in tqdm_e:
            self.curentEpisode = e
            #Start of episode for the parallel A2C actors with their own environment
            futures = [actor.trainOneEpisode.remote() for actor in multiActors]


            # Reset episode
            zeit, cumul_reward, done = 0, 0, False


            env.reset(envLevel[0])# parameter is level
            robotsData = []
            robotsOldState = []
            saveCurrentWeights = False

            for i in range(self.numbOfRobots):

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

                # Actor picks an action (following the policy)
                robotsActions = [] #actions of every Robot in the selected environment
                for i in range(0, len(robotsData)): #iterating over every robot
                    if not True in robotsData[i][3]:
                        aTmp = self.policy_action(robotsOldState[i][0], (reachedTargetList).count(True)/100)
                        a = np.ndarray.tolist(aTmp[0])[0]
                        c = np.ndarray.tolist(aTmp[1])[0]
                    else:
                        a = [None, None]
                    robotsActions.append(a)

                    if not None in a:
                        robotsData[i][0].append(a)#action_onehot) #TODO Tupel mit 2 werten von je -1 bis 1
                        robotsData[i][4].append(c)

                #environment makes a step with selected actions
                results = env.step(robotsActions)


                for i, dataCurrentFrameSingleRobot in enumerate(results):

                    if not True in robotsData[i][3]: #[environment] [robotsData (anstelle von OldState (1)] [Roboter] [done Liste]
                        # print("dataCurent Frame 0 of env",results[j][1], dataCurrentFrame[0])
                        new_state = dataCurrentFrameSingleRobot[0]
                        r = dataCurrentFrameSingleRobot[1]
                        done = dataCurrentFrameSingleRobot[2]
                        robotsData[i][1].append(robotsOldState[i][0])
                        robotsData[i][2].append(r)
                        robotsData[i][3].append(done)
                        if(done):
                            reachedPickup = dataCurrentFrameSingleRobot[3]
                            reachedTargetList.pop(0)
                            reachedTargetList.append(reachedPickup)
                        # Update current state
                        robotsOldState[i] = new_state
                        cumul_reward += r
                zeit += 1

            allTrainingResults = ray.get(futures)
            # if(allTrainingResults )

            if (e+1) % self.args.save_intervall == 0:
                print('Saving')
                self.save_weights(self.args.path)


            allTrainingResults.append(robotsData)
            self.train_models(allTrainingResults)
            trainedWeights = self.network.getWeights()

            for actor in multiActors:
                actor.setWeights.remote(trainedWeights)


            #Checking current success and updating level if nessessary
            allReachedTargetList = reachedTargetList.copy()

            for actor in multiActors:
                tmpTargetList = ray.get(actor.getTargetList.remote())
                allReachedTargetList += tmpTargetList

            targetDivider = (self.numbOfParallelEnvs + 1) * 100  # Erfolg der letzten 100
            successrate = allReachedTargetList.count(True)/targetDivider

            if(successrate>0.75):
                currenthardest = envLevel[0]
                if currenthardest != 8:
                    levelups = self.numbOfParallelEnvs-currenthardest
                    if(self.numbOfParallelEnvs > 20):
                        levelups =  self.numbOfParallelEnvs-2*currenthardest
                    for i in range(levelups): #bei jedem neuen/ schwerern level belibt ein altes level hinten im array aktiv
                        envLevel[i] = envLevel[i]+1

                    print(envLevel)

                    for _ in range(len(reachedTargetList)):
                        reachedTargetList.pop(0)
                        reachedTargetList.append(False)

                    for i, actor in enumerate(multiActors):
                        actor.setLevel.remote(envLevel[i + 1])


            # Update Average Rewards
            self.av_meter.update(cumul_reward)

            # Display score


            tqdm_e.set_description("Reward Episode: " + str(cumul_reward) + " -- Average Reward: " + str(self.av_meter.avg) + " Average Reached Target (last 100): " + str(successrate))
            tqdm_e.refresh()


    def policy_action(self, s, successrate):#TODO obs_timestep mit übergeben
        """ Use the actor to predict the next action to take, using the policy
        """
        # std = ((1-successrate)**2)*0.55


        laser = np.array([np.array(s[i][0]) for i in range(0, len(s))])
        orientation = np.array([np.array(s[i][1]) for i in range(0, len(s))])
        distance = np.array([np.array(s[i][2]) for i in range(0, len(s))])
        velocity = np.array([np.array(s[i][3]) for i in range(0, len(s))])
        timesteps = np.array([np.array(s[i][4]) for i in range(0, len(s))])
        # print(laser.shape, orientation.shape, distance.shape, velocity.shape)
        if(self.timePenalty):
            #Hier breaken um zu gucken, ob auch wirklich 4 timeframes hier eingegeben werden oder was genau das kommt
            return self.network.predict(np.array([laser]), np.array([orientation]), np.array([distance]), np.array([velocity]), np.array([timesteps])) #Liste mit [actions, value]
        else:
            return self.network.predict(np.array([laser]), np.array([orientation]), np.array([distance]), np.array([velocity])) #Liste mit [actions, value]

    def train_models(self, envsData):#, states, actions, rewards): 1 0 2
        """ Update actor and critic networks from experience
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
                    laser = np.array([np.array(s[i][0]) for i in range(0, len(s))])
                    orientation = np.array([np.array(s[i][1]) for i in range(0, len(s))])
                    distance = np.array([np.array(s[i][2]) for i in range(0, len(s))])
                    velocity = np.array([np.array(s[i][3]) for i in range(0, len(s))])
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


                # print("discounted_rewards", discounted_rewards.shape, "state_values", state_values.shape, "advantages",
                #       advantages.shape, "actionsConcatenated", actionsConcatenated.shape, np.vstack(actions).shape)
                # print(len(statesConcatenatedL), len(statesConcatenatedO), len(statesConcatenatedD), len(statesConcatenatedV), len(discounted_rewards), len(actionsConcatenated), len(advantages))

        self.network.train_net(statesConcatenatedL, statesConcatenatedO, statesConcatenatedD,statesConcatenatedV, statesConcatenatedT,discounted_rewards, actionsConcatenated,advantages)

    def discount(self, r):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r = np.zeros_like(r, dtype=float)
        cumul_r = 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def saveCurrentWeights(self):
        print('Saving individual')
        self.save_weights(self.args.path, "_e" + str(self.curentEpisode))

    def save_weights(self, path, additional=""):
        path += 'A2C' + self.args.model_timestamp + additional

        self.network.saveWeights(path)

        data = [self.args]
        with open(path+'.yml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)



    def loadWeights(self, path):
        self.network.load

    def load_weights(self, path):
        self.network.load_weights(path)

        # def load_weights(self, path_actor, path_critic):
        #     self.critic.load_weights(path_critic)
        #     self.actor.load_weights(path_actor)

    def execute(self, env, args):
        robotsCount = self.numbOfRobots

        for e in range(18):

            env.reset(e%9)
            # TODO nach erstem Mal auf trainiertem env sowas wie environment.randomizeForTesting() einbauen (alternativ hier ein fester Testsatz)
            # robotsOldState = [env.get_observation(i) for i in range(0, robotsCount)]
            robotsOldState = [np.expand_dims(env.get_observation(i), axis=0) for i in range(0, robotsCount)]
            robotsDone = [False for i in range(0, robotsCount)]

            while not env.is_done():

                robotsActions = []
                # Actor picks an action (following the policy)
                for i in range(0, robotsCount):

                    if not robotsDone[i]:
                        aTmp = self.network.policy_action_certain(
                            robotsOldState[i][0])  # , (rechedTargetList).count(True) / 100)
                        a = np.ndarray.tolist(aTmp[0])[0]
                    else:
                        a = [None, None]

                    robotsActions.append(a)

                robotsStates= env.step(robotsActions)


                rewards = ''
                for i, stateData in enumerate(robotsStates):
                    new_state = stateData[0]
                    rewards += (str(i) + ': ' + str(stateData[1]) + '   ')
                    done = stateData[2]

                    # if (done):
                    #     reachedPickup = stateData[3]
                    #     rechedTargetList.pop(0)
                    #     rechedTargetList.append(reachedPickup)

                    robotsOldState[i] = new_state
                    if not robotsDone[i]:
                        robotsDone[i] = done
                # print(rewards)


# Bug beim Importieren -> deswegen AverageMeter hierdrin kopiert
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

