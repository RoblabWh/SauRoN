import numpy as np

from algorithms.A2C_parallel.A2C_Multi import AverageMeter
from algorithms.A2C_parallel.PPO_Network import PPO_Network
from algorithms.A2C_parallel.PPO_MultiprocessingActor import PPO_MultiprocessingActor
from tqdm import tqdm
import ray
import yaml



class PPO_Multi:

    def __init__(self, act_dim, env_dim, args):
        self.args = args
        # self.network = PPO_Network(act_dim, env_dim, args)
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.numbOfParallelEnvs = args.parallel_envs
        self.numbOfRobots = args.numb_of_robots
        self.timePenalty = args.time_penalty
        self.av_meter = AverageMeter()
        self.gamma = args.gamma

    def train(self, loadWeightsPath = ""):
        """ Main PPO Training Algorithm
        """
        # reachedTargetList = [False] * 100
        # countEnvs = len(envs)

        if loadWeightsPath != "":
            print("Todo Gewichte für actor laden")

        #Create parallel workers with own environment
        # envLevel = [(i+3)%8 for i in range(self.numbOfParallelEnvs)]
        envLevel = [3 for _ in range(self.numbOfParallelEnvs)]
        ray.init()
        multiActors = [PPO_MultiprocessingActor.remote(self.act_dim, self.env_dim, self.args, None, envLevel[0], True)]
        startweights = multiActors[0].getWeights.remote()
        multiActors += [PPO_MultiprocessingActor.remote(self.act_dim, self.env_dim, self.args, startweights, envLevel[i+1], False) for i in range(self.numbOfParallelEnvs-1)]
        for i, actor in enumerate(multiActors):
            actor.setLevel.remote(envLevel[i])


        # Main Loop
        tqdm_e = tqdm(range(self.args.nb_episodes), desc='Score', leave=True, unit=" episodes")

        for e in tqdm_e:
            self.currentEpisode = e
            currentVar = 0;
            #Start of episode for the parallel PPO actors with their own environment
            #Hier wird die gesamte Episode durchlaufen und dann erst trainiert


            #Training bereits alle n=75 steps mit den zu dem Zeitounkt gesammelten Daten (nur für noch aktive Environments)
            activeActors = multiActors
            while len(activeActors) > 0:
                futures = [actor.trainSteps.remote(self.args.train_interval) for actor in activeActors]
                allTrainingResults = ray.get(futures)
                trainedWeights, var = self.train_models(allTrainingResults, multiActors[0])
                for actor in multiActors[1:len(multiActors)]:
                    actor.setWeights.remote(trainedWeights)
                currentVar = var

                activeActors = []
                for actor in multiActors:
                    if ray.get(actor.isActive.remote()):
                        activeActors.append(actor)



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

            tqdm_e.set_description("R avr last e: " + str(cumul_reward) + " --R avr all e : " + str(self.av_meter.avg) + " --Avr Reached Target (25 epi): " + str(successrate) + " --var: " + str(var[0]))
            tqdm_e.refresh()


    def train_models(self, envsData, masterEnv = None):#, states, actions, rewards): 1 0 2
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
        self.save_weights(self.args.path, "_e" + str(self.currentEpisode))

    def save_weights(self, masterEnv, path, additional=""):
        path += 'PPO' + self.args.model_timestamp + additional

        masterEnv.saveWeights.remote(path)

        data = [self.args]
        with open(path+'.yml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)



    def loadWeights(self, path):
        self.network.load

    def load_net(self, path):
        self.network = PPO_Network(self.act_dim, self.env_dim, self.args)
        self.network.load_weights(path)

    def execute(self, env, args):

        robotsCount = self.numbOfRobots

        for e in range(18):

            env.reset(e % 8)
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
                        a = np.ndarray.tolist(aTmp[0])
                    else:
                        a = [None, None]

                    robotsActions.append(a)

                robotsStates = env.step(robotsActions)

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