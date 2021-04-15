import numpy as np


from algorithms.A2C_parallel.A2C_Network import A2C_Network
from algorithms.A2C_parallel.A2C_MultiprocessingActor import A2C_MultiprocessingActor
from tqdm import tqdm
import ray
import yaml



class A2C_Multi:

    def __init__(self, act_dim, env_dim, args):
        self.args = args
        # self.network = A2C_Network(act_dim, env_dim, args)
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.numbOfParallelEnvs = args.parallel_envs
        self.numbOfRobots = args.numb_of_robots
        self.timePenalty = args.time_penalty
        self.av_meter = AverageMeter()
        self.gamma = args.gamma

    def train(self):
        """ Main A2C Training Algorithm
        """
        # reachedTargetList = [False] * 100
        # countEnvs = len(envs)
        envLevel = [0 for _ in range(self.numbOfParallelEnvs)]

        ray.init()
        multiActors = [A2C_MultiprocessingActor.remote(self.act_dim, self.env_dim, self.args, None, envLevel[0], True)]
        startweights = multiActors[0].getWeights.remote()
        multiActors += [A2C_MultiprocessingActor.remote(self.act_dim, self.env_dim, self.args, startweights, envLevel[i+1], False) for i in range(self.numbOfParallelEnvs-1)]
        for i, actor in enumerate(multiActors):
            actor.setLevel.remote(envLevel[i])
        # env.setUISaveListener(self)

        # Main Loop
        tqdm_e = tqdm(range(self.args.nb_episodes), desc='Score', leave=True, unit=" episodes")

        for e in tqdm_e:
            self.currentEpisode = e
            #Start of episode for the parallel A2C actors with their own environment
            # HIer wird die gesamte Episode durchlaufen und dann erst trainiert
            # futures = [actor.trainOneEpisode.remote() for actor in multiActors]


            #Training bereits alle n=75 Episoden mit den zu dem Zeitounkt gesammelten Daten (nur für noch aktive Environments)

            activeActors = multiActors
            while len(activeActors) > 0:
                futures = [actor.trainSteps.remote(75) for actor in activeActors]

                allTrainingResults = ray.get(futures)
                trainedWeights = self.train_models(allTrainingResults, multiActors[0])
                for actor in multiActors[1:len(multiActors)]:
                    actor.setWeights.remote(trainedWeights)

                activeActors = []
                for actor in multiActors:
                    if ray.get(actor.isActive.remote()):
                        activeActors.append(actor)
                # activeActors = [actor for actor in multiActors if ray.get(actor.isActive.remote())]

            for actor in multiActors:
                actor.reset.remote()




            # Reset episode
            zeit, cumul_reward, done = 0, 0, False
            #TODO Werte für ausgabe aus Actorn ziehen


            #Benötigt für Durchlauf ganzer Episode mit Training am Ende
            # allTrainingResults = ray.get(futures)

            if (e+1) % self.args.save_intervall == 0:
                print('Saving')
                self.save_weights(multiActors[0], self.args.path)

            #Checking current success and updating level if nessessary
            # allReachedTargetList = reachedTargetList.copy()

            allReachedTargetList = []

            for actor in multiActors:
                tmpTargetList = ray.get(actor.getTargetList.remote())
                allReachedTargetList += tmpTargetList

            # allTrainingResults.append(robotsData)

            targetDivider = (self.numbOfParallelEnvs) * 100  # Erfolg der letzten 100
            successrate = allReachedTargetList.count(True) / targetDivider

            if (successrate > 0.83):
                lastindex = len(allReachedTargetList)
                currenthardest = envLevel[0]
                if currenthardest != 8:
                    levelups = self.numbOfParallelEnvs - (currenthardest+1)
                    if (self.numbOfParallelEnvs > 20):
                        levelups = self.numbOfParallelEnvs - 2 * currenthardest+1
                    for i in range(levelups):  # bei jedem neuen/ schwerern level belibt ein altes level hinten im array aktiv
                        envLevel[i] = envLevel[i] + 1

                    print(envLevel)
                    self.save_weights(multiActors[0], self.args.path, "_endOfLevel-"+str(currenthardest))


                    for i, actor in enumerate(multiActors):
                        actor.setLevel.remote(envLevel[i])

            #Benötigt für Durchlauf ganzer Episode mit Training am Ende
            # trainedWeights = self.train_models(allTrainingResults, multiActors[0])
            # trainedWeights = self.network.getWeights()

            #Benötigt für Durchlauf ganzer Episode mit Training am Ende
            # for actor in multiActors[1:len(multiActors)]:
            #     actor.setWeights.remote(trainedWeights)

            # Update Average Rewards
            self.av_meter.update(cumul_reward)

            # Display score


            tqdm_e.set_description("Epi r: " + str(cumul_reward) + " -- Avr r: " + str(self.av_meter.avg) + " Avr Reached Target (25 epi): " + str(successrate))
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

    def train_models(self, envsData, masterEnv):#, states, actions, rewards): 1 0 2
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

        weights = ray.get(masterEnv.trainNet.remote(statesConcatenatedL, statesConcatenatedO, statesConcatenatedD,statesConcatenatedV, statesConcatenatedT,discounted_rewards, actionsConcatenated,advantages))
        # self.network.train_net(statesConcatenatedL, statesConcatenatedO, statesConcatenatedD,statesConcatenatedV, statesConcatenatedT,discounted_rewards, actionsConcatenated,advantages)
        return weights

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
        path += 'A2C' + self.args.model_timestamp + additional

        masterEnv.saveWeights.remote(path)

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

