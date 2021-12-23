import numpy as np
import keras as k

from tqdm import tqdm
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Lambda, Conv1D, concatenate
from keras.optimizers import RMSprop, Adam
from keras.losses import mean_squared_error
from keras.layers import Input, Conv1D, Dense, Flatten, concatenate, MaxPool1D, Lambda
from keras.backend import max, mean, exp, log, function, squeeze, categorical_crossentropy,placeholder, sum, square, random_normal, shape, cast, clip, softmax, argmax
from keras import backend as K
import ray
import time
import datetime
import multiprocessing
import concurrent.futures


#from utils import AverageMeter
@ray.remote
def stepSingleEnv(env, actions):
    return (env.step(actions), env)

def stepSingleEnvPool(env, actions):
    return env.step(actions)

class A2C_C:
    """ Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, args):
        """ Initialization
        """
        ray.init()
        print(k.__version__)
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.gamma = args.gamma
        self.lr = args.learningrate
        self.rays = int(360/args.angle_steps)
        self.timePenalty = args.time_penalty

        self._input_laser = Input(shape=(4, self.rays), dtype='float32', name='input_laser')
        # Orientation input
        self._input_orientation = Input(shape=(4, 2,), dtype='float32', name='input_orientation')
        # Distance input
        self._input_distance = Input(shape=(4, 1,), dtype='float32', name='input_distance')
        # Velocity input
        self._input_velocity = Input(shape=(4, 2,), dtype='float32', name='input_velocity')
        # Passed Time input
        self._input_timestep = Input(shape=(4, 1,), dtype='float32', name='input_Timestep')

        # Create actor and critic networks
        self.buildNetWithOpti()

        self.av_meter = AverageMeter()
        self.args = args


    def buildNetWithOpti(self):
        self._ADVANTAGE = placeholder(shape=(None,), name='ADVANTAGE')
        self._REWARD = placeholder(shape=(None,), name='REWARD')
        self._ACTION = placeholder(shape=(None, 2), name='ACTION')




        #
        # # Laser input und convolutions
        # # x_laser = Conv1D(filters=16, kernel_size=7, strides=3, padding='same', activation='relu',
        # #                  name='shared' + '_conv1d_laser_1')(self._input_laser)
        # # x_laser = Conv1D(filters=32, kernel_size=5, strides=2, padding='same', activation='relu',
        # #                  name='shared' + '_conv1d_laser_2')(x_laser)
        # x_laserf = Flatten()(self._input_laser)
        #
        # x_laser = Dense(units=256, activation='relu', name='shared' + '_dense_laser')(x_laserf)
        #
        # # Orientation input
        # x_orientationf = Flatten()(self._input_orientation)
        # x_orientation = Dense(units=32, activation='relu', name='shared' + '_dense_orientation')(x_orientationf)
        #
        # # Distance input
        # x_distancef = Flatten()(self._input_distance)
        # x_distance = Dense(units=16, activation='relu', name='shared' + '_dense_distance')(x_distancef)
        #
        # # Velocity input
        # x_velocityf = Flatten()(self._input_velocity)
        # x_velocity = Dense(units=32, activation='relu', name='shared' + '_dense_velocity')(x_velocityf)
        #
        # # Fully connect
        # fully_connect = concatenate([x_laser, x_orientation, x_distance, x_velocity])
        # fully_connect = Dense(units=384, activation='relu', name='shared' + '_dense_fully_connect')(fully_connect)

        x_laser = Conv1D(filters=12, kernel_size=5, strides=3, padding='same', activation='relu',
                         name='shared' + '_conv1d_laser_1')(self._input_laser)
        x_laser = Conv1D(filters=24, kernel_size=3, strides=2, padding='same', activation='relu',
                         name='shared' + '_conv1d_laser_2')(x_laser)
        x_laser = Flatten()(x_laser)

        x_laser = Dense(units=192, activation='relu', name='shared' + '_dense_laser')(x_laser)

        # Orientation input
        x_orientation = Flatten()(self._input_orientation)

        # Distance input
        x_distance = Flatten()(self._input_distance)

        # Velocity input
        x_velocity = Flatten()(self._input_velocity)

        # (passed) Timestep input
        x_timestep = Flatten()(self._input_timestep)

        if self.timePenalty:
            concated0 = concatenate([x_orientation, x_distance, x_velocity, x_timestep])
        else:
            concated0 = concatenate([x_orientation, x_distance, x_velocity])

        concated = Dense(units=64, activation='relu', name='shared' + '_dense_concated')(concated0)

        # Fully connect
        fully_connect = concatenate([x_laser, concated])
        fully_connect = Dense(units=384, activation='relu', name='shared' + '_dense_fully_connect')(fully_connect)


        self._mu_var = ContinuousLayer()(fully_connect)
        self._mu = Lambda(lambda x: x[:, :2])(self._mu_var)
        self._var = Lambda(lambda x: x[:, -2:])(self._mu_var)
        self._selected_action = self.select_action_continuous_clip(self._mu, self._var)
        self._neglog = self.neglog_continuous(self._selected_action, self._mu, self._var)
        self._neglogp = self.neglog_continuous(self._ACTION, self._mu, self._var)

        pg_loss = mean(self._ADVANTAGE * self._neglogp)


        #critic
        self._value = Dense(units=1, activation='linear', name='value')(fully_connect)

        value_loss = mean_squared_error(squeeze(self._value, axis=-1), self._REWARD) * 0.5

        # entropy
        entropy = self.entropy_continuous(self._var)
        entropy = mean(entropy) * 0.1

        loss = pg_loss + value_loss - entropy

        if self.timePenalty:
            self._model = Model(
                inputs=[self._input_laser, self._input_orientation, self._input_distance, self._input_velocity, self._input_timestep],
                outputs=[self._mu, self._var, self._value])
        else:
            self._model = Model(
                inputs=[self._input_laser, self._input_orientation, self._input_distance, self._input_velocity],
                outputs=[self._mu, self._var, self._value])

        # Optimizer
        self._optimizer = Adam(lr=self.lr, epsilon=1e-5, clipnorm=1.0)

        updates = self._optimizer.get_updates(self._model.trainable_weights, [], loss)
        if self.timePenalty:
            self._train = function(
                [self._input_laser, self._input_orientation, self._input_distance, self._input_velocity, self._input_timestep, self._REWARD,
                 self._ACTION, self._ADVANTAGE], [loss, pg_loss, value_loss, entropy], updates)

            self._predict = function(
                [self._input_laser, self._input_orientation, self._input_distance, self._input_velocity, self._input_timestep],
                [self._selected_action, self._value])
            self._sample = function(
                [self._input_laser, self._input_orientation, self._input_distance, self._input_velocity, self._input_timestep], [self._mu])
        else:
            self._train = function(
                [self._input_laser, self._input_orientation, self._input_distance, self._input_velocity,
                 self._REWARD, self._ACTION, self._ADVANTAGE], [loss, pg_loss, value_loss, entropy], updates)

            self._predict = function(
                [self._input_laser, self._input_orientation, self._input_distance, self._input_velocity],
                [self._selected_action, self._value])
            self._sample = function(
                [self._input_laser, self._input_orientation, self._input_distance, self._input_velocity], [self._mu])
        # self._model.summary()

    def select_action_continuous_clip(self, mu, var):
        return clip(mu + exp(var) * random_normal(shape(mu)), -1.0, 1.0)

    def neglog_continuous(self, action, mu, var):
        return 0.5 * sum(square((action - mu) / exp(var)), axis=-1) \
                + 0.5 * log(2.0 * np.pi) * cast(shape(action)[-1], dtype='float32') \
                + sum(var, axis=-1)

    def entropy_continuous(self, var):
        return sum(var + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)



    def train_net(self, obs_laser, obs_orientation_to_goal, obs_distance_to_goal, obs_velocity, obs_timestep, rewards, actions, advantage):
        if self.timePenalty:
            loss, pg_loss, value_loss, entropy = self._train([obs_laser, obs_orientation_to_goal, obs_distance_to_goal, obs_velocity, obs_timestep, rewards, actions, advantage])
        else:
            loss, pg_loss, value_loss, entropy = self._train([obs_laser, obs_orientation_to_goal, obs_distance_to_goal, obs_velocity, rewards, actions, advantage])

        return loss, pg_loss, value_loss, entropy


    def predict(self, obs_laser, obs_orientation_to_goal, obs_distance_to_goal, obs_velocity, obs_timestep): #!!!!!!!!!!!!!!!die neue policy_action
        #samples, values, neglogs, mu, var = self._predict([obs_laser, obs_orientation_to_goal, obs_distance_to_goal])
        action, values = self._predict([obs_laser, obs_orientation_to_goal, obs_distance_to_goal, obs_velocity, obs_timestep])

        return action, values


    def policy_action(self, s, successrate):
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
            return self.predict(np.array([laser]), np.array([orientation]), np.array([distance]), np.array([velocity]), np.array([timesteps])) #Liste mit [actions, value]
        else:
            return self._predict([np.array([laser]), np.array([orientation]), np.array([distance]), np.array([velocity])]) #Liste mit [actions, value]



    def policy_action_certain(self, s):
        """ Use the actor to predict the next action to take, using the policy
        """
        # std = ((1-successrate)**2)*0.55


        laser = np.array([np.array(s[i][0]) for i in range(0, len(s))])
        orientation = np.array([np.array(s[i][1]) for i in range(0, len(s))])
        distance = np.array([np.array(s[i][2]) for i in range(0, len(s))])
        velocity = np.array([np.array(s[i][3]) for i in range(0, len(s))])
        # timesteps = np.array([np.array(0) for i in range(0, len(s))])
        timesteps = np.array([np.array(s[i][4]) for i in range(0, len(s))])
        if(self.timePenalty):
            mu = self._sample([np.array([laser]),  np.array([orientation]), np.array([distance]), np.array([velocity]), np.array([timesteps])])
        else:
            mu = self._sample([np.array([laser]),  np.array([orientation]), np.array([distance]), np.array([velocity])])

        return mu


    def discount(self, r):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r = np.zeros_like(r, dtype=float)
        cumul_r = 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

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
            for data in robotsData[0]:
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

        self.train_net(statesConcatenatedL, statesConcatenatedO, statesConcatenatedD,statesConcatenatedV, statesConcatenatedT,discounted_rewards, actionsConcatenated,advantages)



        # self.actor.fit(statesConcatenated, actionsConcatenated, sample_weight=advantages, epochs=1, verbose=0)
        # self.critic.fit(statesConcatenated, discounted_rewards, epochs=1, verbose=0)

    def train(self, envs, args):
        """ Main A2C Training Algorithm
        """
        self.taktischeZeit = datetime.datetime.now().strftime("%d%H%M%b%y")#Zeitstempel beim Start des trainings für das gespeicherte Modell
        results = []            # wird nirgendwo gebraucht -> returned leeres Array

        liste = np.array([], dtype=object)
        # Main Loop
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")
        waitForN = 10
        rechedTargetList = [False] * 100
        countRobots = 4
        countEnvs = len(envs)



        for e in tqdm_e:

            # Reset episode
            zeit, cumul_reward, done = 0, 0, False


            envsData = []

            for j in range(len(envs)):
                envs[j].reset()
                robotsData = []
                robotsOldState = []

                for i in range(countRobots):

                    old_state = envs[j].get_observation(i)
                    robotsOldState.append(np.expand_dims(old_state, axis=0))


                    actions, states, rewards, done, evaluation = [], [], [], [], []
                    robotsData.append((actions, states, rewards, done, evaluation))
                # Robot 0 actions --> robotsData[0][0]
                # Robot 0 states  --> robotsData[0][1]
                # Robot 0 rewards --> robotsData[0][2]
                # Robot 1 actions --> robotsData[1][0]
                # ...
                envsData.append((robotsData, robotsOldState))

            allDone = False
            while not allDone:

                envActions = []
                for j in range(0, len(envsData)):
                # for singleEnvData in envsData:
                    robotsActions = [] #actiobs for every Robot in the selected environment
                    # Actor picks an action (following the policy)
                    for i in range(0, len(envsData[j][0])): #iterating over every robot
                        if not True in envsData[j][0][i][3]:
                            # a = self.predict(singleEnvData[1][i][0:90][:], )
                            aTmp = self.policy_action(envsData[j][1][i][0], (rechedTargetList).count(True)/100)
                            a = np.ndarray.tolist(aTmp[0])[0]
                            c = np.ndarray.tolist(aTmp[1])[0]
                            # print(a,c)
                        else:
                            a = [None, None]
                        # action_onehot = np.zeros([self.act_dim])
                        # action_onehot[a] = 1

                        robotsActions.append(a)

                        if not None in a:
                            envsData[j][0][i][0].append(a)#action_onehot)
                            envsData[j][0][i][4].append(c)

                    # Retrieve new state, reward, and whether the state is terminal
                    # new_state, r, done = env.step(robotsActions)

                    envActions.append(robotsActions)

                time1 = time.time()
                #### Multiprocessing ####
                # with concurrent.futures.ThreadPoolExecutor() as executor:
                #    results = executor.map(stepSingleEnv, envs, envActions)

                #### ohne Multiprocessing ####
                # results = []
                # for j, env in enumerate(envs):
                #     results.append(env.step(envActions[j]))


                #### Multiprocessing mit Ray ####
                # futures = [stepSingleEnv.remote(envs[j], envActions[j]) for j in range(countEnvs-1)]
                futures = [stepSingleEnv.remote(envs[j], envActions[j]) for j in range(countEnvs-1)]
                resultsEnvUI = envs[countEnvs-1].step(envActions[countEnvs-1])
                returnValues = ray.get(futures)
                results = []
                envTmp = []
                for tuple in returnValues:
                    results.append(tuple[0])
                    envTmp.append(tuple[1])
                envTmp.append(envs[countEnvs-1])
                envs = envTmp
                results.append(resultsEnvUI)

                #robotsDataCurrentFrame = env.step(robotsActions)

                #print("reward " + str(r))
                # Memorize (s, a, r) for training
                resultList = []
                for result in results:
                    #print(result)
                    resultList.append(result[0])
                    # print(result[1])
                time2 = time.time()
                #print("Process Time", time2-time1)
                #print(resultList)#.sort(key=lambda x:x[1]))

                for j, robotsDataCurrentFrameSingleEnv in enumerate(resultList):

                    for i, dataCurrentFrameSingleRobot in enumerate(robotsDataCurrentFrameSingleEnv):

                        if not True in envsData[j][0][i][3]: #[environment] [robotsData (anstelle von OldState (1)] [Roboter] [done Liste]
                            # print("dataCurent Frame 0 of env",results[j][1], dataCurrentFrame[0])
                            new_state = dataCurrentFrameSingleRobot[0]
                            r = dataCurrentFrameSingleRobot[1]
                            done = dataCurrentFrameSingleRobot[2]
                            envsData[j][0][i][1].append(envsData[j][1][i][0])
                            envsData[j][0][i][2].append(r)
                            envsData[j][0][i][3].append(done)
                            if(done):
                                reachedPickup = dataCurrentFrameSingleRobot[3]
                                rechedTargetList.pop(0)
                                rechedTargetList.append(reachedPickup)
                            # Update current state
                            envsData[j][1][i] = new_state
                            cumul_reward += r
                    # print("Kumulierter Reward: " + str(cumul_reward) + ", Reward: " + str(r))
                zeit += 1
                #print(zeit)
                allDone = True
                for j, env in enumerate(envs):
                    # print(j, env.is_done(), zeit)
                    if not env.is_done():
                        allDone = False
                    else:
                        if not j in debugFertigesEnv:
                            print(j, ' done!', zeit)
                            debugFertigesEnv.append(j)



            self.train_models(envsData)

            if e % args.save_intervall == 0:
                print('Saving')
                self.save_weights(args.path)

            # Update Average Rewards
            self.av_meter.update(cumul_reward)

            # Display score
            tqdm_e.set_description("Reward Episode: " + str(cumul_reward) + " -- Average Reward: " + str(self.av_meter.avg) + " Average Reached Target (last 100): " + str((rechedTargetList).count(True)/100))
            tqdm_e.refresh()

        return results

    def save_weights(self, path):
        path += 'A2C'
        self._model.save_weights(path + '_actor_Critic_' + self.args.mode + self.taktischeZeit +'.h5')

    def load_weights(self, path):
        self._model.load_weights(path)

   # def load_weights(self, path_actor, path_critic):
   #     self.critic.load_weights(path_critic)
   #     self.actor.load_weights(path_actor)

    def execute(self, env, args):
        robotsCount = 4

        for e in range (0,4):

            env.reset()

            robotsOldState = [np.expand_dims(env.get_observation(i), axis=0) for i in range(0, robotsCount)]
            robotsDone = [False for i in range(0, robotsCount)]

            while not env.is_done():

                robotsActions = []
                # Actor picks an action (following the policy)
                for i in range(0, robotsCount):

                    if not robotsDone[i]:
                        aTmp = self.policy_action_certain(robotsOldState[i][0])#, (rechedTargetList).count(True) / 100)
                        a = np.ndarray.tolist(aTmp[0])[0]
                    else:
                        a = [None, None]

                    robotsActions.append(a)


                robotsStates = env.step(robotsActions)[0]

                rewards = ''
                for i, stateData in enumerate(robotsStates):
                    new_state = stateData[0]
                    rewards += (str(i)+': '+str(stateData[1]) + '   ')
                    done = stateData[2]

                    # if (done):
                    #     reachedPickup = stateData[3]
                    #     rechedTargetList.pop(0)
                    #     rechedTargetList.append(reachedPickup)

                    robotsOldState[i] = new_state
                    if not robotsDone[i]:
                        robotsDone[i]= done
                print(rewards)



from keras.layers import Layer, Dense, Input, concatenate, InputSpec
from keras.models import Model
import keras as K

class ContinuousLayer(Layer):
    def __init__(self, **kwargs):
        self._mu = Dense(units=2, activation='tanh', name='mu', kernel_initializer=K.initializers.Orthogonal(gain=1), use_bias=True, bias_initializer='zero')
        super(ContinuousLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self._var = self.add_weight(name='kernel',
                                    shape=(2,),
                                    initializer='zero',
                                    trainable=True)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, x, **kwargs):
        tmp = self._mu(x)
        return concatenate([tmp, tmp * 0.0 + self._var], axis=-1)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = 2
        return tuple(output_shape)

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


import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

