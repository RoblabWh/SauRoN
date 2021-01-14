import numpy as np
import keras as k
import copy
import concurrent.futures

from tqdm import tqdm
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Lambda, Conv1D, concatenate
from keras.optimizers import RMSprop, Adam
from keras.losses import mean_squared_error
from keras.layers import Input, Conv1D, Dense, Flatten, concatenate, MaxPool1D, Lambda
from keras.backend import max, mean, exp, log, function, squeeze, categorical_crossentropy,placeholder, sum, square, random_normal, shape, cast, clip, softmax, argmax
from keras import backend as K

class A2C_Network:
    def __init__(self, act_dim, env_dim, args):
        """ Initialization """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.gamma = args.gamma
        self.lr = args.learningrate
        self.rays = int(360 / args.angle_steps)
        self.timePenalty = args.time_penalty

        self._input_laser = Input(shape=(4, self.rays), dtype='float32', name='input_laser')
        # Orientation input
        self._input_orientation = Input(shape=(4, 2,), dtype='float32', name='input_orientation')
        # Distance input
        self._input_distance = Input(shape=(4, 1,), dtype='float32', name='input_distance')
        # Velocity input
        self._input_velocity = Input(shape=(4, 2,), dtype='float32', name='input_velocity')
        # Passed Time input
        self._input_timestep = Input(shape=(1, 1,), dtype='float32', name='input_Timestep')

        # Create actor and critic networks
        self.buildNetWithOpti()

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

        # critic
        self._value = Dense(units=1, activation='linear', name='value')(fully_connect)

        value_loss = mean_squared_error(squeeze(self._value, axis=-1), self._REWARD) * 0.5

        # entropy
        entropy = self.entropy_continuous(self._var)
        entropy = mean(entropy) * 0.1

        loss = pg_loss + value_loss - entropy

        if self.timePenalty:
            self._model = Model(
                inputs=[self._input_laser, self._input_orientation, self._input_distance, self._input_velocity,
                        self._input_timestep],
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
                [self._input_laser, self._input_orientation, self._input_distance, self._input_velocity,
                 self._input_timestep, self._REWARD,
                 self._ACTION, self._ADVANTAGE], [loss, pg_loss, value_loss, entropy], updates)

            self._predict = function(
                [self._input_laser, self._input_orientation, self._input_distance, self._input_velocity,
                 self._input_timestep],
                [self._selected_action, self._value])
            self._sample = function(
                [self._input_laser, self._input_orientation, self._input_distance, self._input_velocity,
                 self._input_timestep], [self._mu])
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

    def policy_action(self, s, successrate):#TODO obs_timestep mit übergeben
        """ Use the actor to predict the next action to take, using the policy
        """
        # std = ((1-successrate)**2)*0.55


        laser = np.array([np.array(s[i][0]) for i in range(0, len(s))])
        orientation = np.array([np.array(s[i][1]) for i in range(0, len(s))])
        distance = np.array([np.array(s[i][2]) for i in range(0, len(s))])
        velocity = np.array([np.array(s[i][3]) for i in range(0, len(s))])
        timesteps = np.array([np.array(s[3][4])])
        # print(laser.shape, orientation.shape, distance.shape, velocity.shape)
        if(self.timePenalty):
            return self.predict(np.array([laser]), np.array([orientation]), np.array([distance]), np.array([velocity]), np.array([timesteps])) #Liste mit [actions, value]
        else:
            return self._predict([np.array([laser]), np.array([orientation]), np.array([distance]), np.array([velocity])]) #Liste mit [actions, value]

    def discount(self, r):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r = np.zeros_like(r, dtype=float)
        cumul_r = 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def train_models(self, robotsData):#, states, actions, rewards): 1 0 2
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
                usedTimeStep = np.array([np.array(s[3][4])])
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

    def getWeights(self):
        return self._model.get_weights()



######################################################
############# Helper Class for NNet ##################
######################################################

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

