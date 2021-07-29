import numpy as np
import keras as k

from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.layers import Input, Conv1D, Dense, Flatten, concatenate, MaxPool1D, Lambda
from keras.backend import maximum, mean, exp, log, function, squeeze, categorical_crossentropy,placeholder, sum, square, random_normal, shape, cast, clip, softmax, argmax, gradients
from keras import backend as K




class PPO_Network:
    """
    Neural Network for Actor Critic
    """

    def __init__(self, act_dim, env_dim, args):
        """ Initialization
        """
        print(k.__version__)
        # Environment and A2C parameters
        self.args = args
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.gamma = args.gamma
        self.lr = args.learningrate
        self.rays = args.number_of_rays
        self.timePenalty = args.time_penalty

        self._network_size = args.net_size
        self._shared = args.shared
        self._coefficient_value = 0.5
        self._coefficient_entropy = 0.0
        self._clip_range = 0.2

        self._input_laser = Input(shape=(self.rays, env_dim[0]), dtype='float32', name='input_laser')
        # Orientation input
        self._input_orientation = Input(shape=(2, env_dim[0],), dtype='float32', name='input_orientation')
        # Distance input
        self._input_distance = Input(shape=(1, env_dim[0], ), dtype='float32', name='input_distance')
        # Velocity input
        self._input_velocity = Input(shape=(2,env_dim[0],), dtype='float32', name='input_velocity')
        # Passed Time input
        self._input_timestep = Input(shape=(1,env_dim[0],), dtype='float32', name='input_Timestep')

        # Create actor and critic networks
        self.buildNetWithOpti()

    def buildMainNet(self, tag = 'shared', type = 'big'):
        """
        builds the part of the neural network, that  processes all input values into one fully connected layer
        with a variety of processing depending on the chosen type.
        :param tag: String - Is used to name the layers for easier identification when using something like summary().
        :param type: String - Choose ["small", "medium", "big"] to determine the amount of layers and convolutions used.
        :return: returns the final layer of the created network which can be used as a new input
        """

        if type == 'big':
            # Laser input und convolutions
            x_laser = Conv1D(filters=16, kernel_size=7, strides=3, padding='same', activation='relu',
                             name=tag + '_conv1d_laser_1')(self._input_laser)
            x_laser = Conv1D(filters=32, kernel_size=5, strides=2, padding='same', activation='relu',
                             name=tag + '_conv1d_laser_last')(x_laser)

            x_laser = Flatten()(x_laser)

            x_laser = Dense(units=256, activation='relu', name=tag + '_dense_laser')(x_laser)

            # Orientation input
            x_orientation = Flatten()(self._input_orientation)
            x_orientation = Dense(units=32, activation='relu', name=tag + '_dense_orientation')(x_orientation)

            # Distance input
            x_distance = Flatten()(self._input_distance)
            x_distance = Dense(units=16, activation='relu', name=tag + '_dense_distance')(x_distance)

            # Velocity input
            x_velocity = Flatten()(self._input_velocity)
            x_velocity = Dense(units=32, activation='relu', name=tag + '_dense_velocity')(x_velocity)

            # Fully connect
            fully_connect = concatenate([x_laser, x_orientation, x_distance, x_velocity])
            fully_connect = Dense(units=384, activation='relu', name=tag + '_dense_fully_connect')(fully_connect)




            return fully_connect

        elif type == 'medium':
            # Laser input und convolutions
            x_laser = Conv1D(filters=12, kernel_size=7, strides=3, padding='same', activation='relu',
                             name=tag + '_conv1d_laser_1')(self._input_laser)
            x_laser = Conv1D(filters=24, kernel_size=5, strides=2, padding='same', activation='relu',
                             name=tag + '_conv1d_laser_last')(x_laser)

            x_laser = Flatten()(x_laser)

            x_laser = Dense(units=192, activation='relu', name=tag + '_dense_laser')(x_laser)

            # Orientation input
            x_orientation = Flatten()(self._input_orientation)

            # Distance input
            x_distance = Flatten()(self._input_distance)

            # Velocity input
            x_velocity = Flatten()(self._input_velocity)

            concated = concatenate([x_orientation, x_distance, x_velocity])
            concated = Dense(units=64, activation='relu', name=tag + '_dense_concated')(concated)

            # Fully connect
            fully_connect = concatenate([x_laser, concated])
            fully_connect = Dense(units=384, activation='relu', name=tag + '_dense_fully_connect')(fully_connect)

            return fully_connect

        elif type == 'small':
            # Laser input und convolutions
            x_laser = Conv1D(filters=8, kernel_size=7, strides=3, padding='same', activation='relu',
                             name=tag + '_conv1d_laser_1')(self._input_laser)
            x_laser = Conv1D(filters=16, kernel_size=5, strides=2, padding='same', activation='relu',
                             name=tag + '_conv1d_laser_last')(x_laser)

            x_laser = Flatten()(x_laser)

            x_laser = Dense(units=128, activation='relu', name=tag + '_dense_laser')(x_laser)

            # Orientation input
            x_orientation = Flatten()(self._input_orientation)

            # Distance input
            x_distance = Flatten()(self._input_distance)

            # Velocity input
            x_velocity = Flatten()(self._input_velocity)

            # Fully connect
            fully_connect = concatenate([x_laser, x_orientation, x_distance, x_velocity])
            fully_connect = Dense(units=128, activation='relu', name=tag + '_dense_fully_connect')(fully_connect)

            return fully_connect

        else:
            print("Network type ", str(type), " is unknown! select from: small, medium or big!")
            exit(1)

    def buildNetWithOpti(self):
        """
        constructs the neural network and defines the optimizer with its loss function
        """
        self._ADVANTAGE = placeholder(shape=(None,), name='ADVANTAGE')
        self._REWARD = placeholder(shape=(None,), name='REWARD')
        self._ACTION = placeholder(shape=(None, 2), name='ACTION')
        self._OLD_NEGLOG = placeholder(shape=(None,), name='OLD_NEGLOG')
        self._NEGLOG = placeholder(shape=(None,), name='NEGLOG')


        fully_connect = self.buildMainNet("shared" if self._shared else "policy", self._network_size)

        # actor
        self._mu_var = ContinuousLayer()(fully_connect)
        self._mu = Lambda(lambda x: x[:, :2])(self._mu_var)
        self._var = Lambda(lambda x: x[:, -2:])(self._mu_var)
        self._selected_action = self.select_action_continuous_clip(self._mu, self._var)
        self._neglog = self.neglog_continuous(self._selected_action, self._mu, self._var)
        self._neglogp = self.neglog_continuous(self._ACTION, self._mu, self._var)
        self._sample = self._mu

        ratio = exp(self._OLD_NEGLOG - self._neglogp)
        pg_loss1 = - self._ADVANTAGE * ratio
        pg_loss2 = - self._ADVANTAGE * clip(ratio, 1.0 - self._clip_range, 1.0 + self._clip_range)

        pg_loss = mean(maximum(pg_loss1, pg_loss2))

        # critic
        if self._shared:
            fully_connect2 = fully_connect
        else:
            fully_connect2 = self.buildMainNet('value', self._network_size)
        self._value = Dense(units=1, activation='linear')(fully_connect2)


        value_loss = mean_squared_error(squeeze(self._value, axis=-1), self._REWARD) * self._coefficient_value

        # entropy
        entropy = self.entropy_continuous(self._var)
        entropy = mean(entropy) * self._coefficient_entropy

        loss = pg_loss + value_loss - entropy

        # build model
        self._model = Model(
            inputs=[self._input_laser, self._input_orientation, self._input_distance, self._input_velocity],
            outputs=[self._mu, self._var, self._value])

        outputLin = [self._model.output[0][:, 0]] #, self._model.output[1][0], self._model.output[2]]
        outputAng = [self._model.output[0][:, 1]] #, self._model.output[1][1], self._model.output[2]]
        lastConvLayer = self._model.get_layer("shared" if self._shared else "policy" + '_conv1d_laser_last')

        grads = gradients(outputLin, lastConvLayer.output)[0]
        print("Grads: ", grads)

        # This is a vector of shape (512,), where each entry
        # is the mean intensity of the gradient over a specific feature map channel
        pooled_grads = mean(grads, axis=(0,1))
        print("Pooled Grads shape: ", pooled_grads)

        # This function allows us to access the values of the quantities we just defined:
        # `pooled_grads` and the output feature map of `block5_conv3`,
        # given a sample image
        self.iterate = function([self._model.input], [pooled_grads, lastConvLayer.output[0], self._model.output])



        # Optimizer
        self._optimizer = Adam(lr=self.lr, epsilon=1e-5, clipnorm=1.0)
        updates = self._optimizer.get_updates(self._model.trainable_weights, [], loss)

        self._train = function(
            [self._input_laser, self._input_orientation, self._input_distance, self._input_velocity, self._REWARD,
             self._ACTION, self._ADVANTAGE, self._OLD_NEGLOG], [loss, pg_loss, value_loss, entropy, self._var], updates)


        self._predict = function(
            [self._input_laser, self._input_orientation, self._input_distance, self._input_velocity],
            [self._selected_action, self._value, self._neglog])
        self._sample = function(
            [self._input_laser, self._input_orientation, self._input_distance, self._input_velocity], self._mu)

        #self.printSummary()


    def select_action_continuous_clip(self, mu, var):
        return clip(mu + exp(var) * random_normal(shape(mu)), -1.0, 1.0)

    def neglog_continuous(self, action, mu, var):
        return 0.5 * sum(square((action - mu) / exp(var)), axis=-1) \
                + 0.5 * log(2.0 * np.pi) * cast(shape(action)[-1], dtype='float32') \
                + sum(var, axis=-1)

    def entropy_continuous(self, var):
        return sum(var + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def printSummary(self):
        self._model.summary()

    def load(self, path):
        self._model.load_weights(path)



    def train_net(self, obs_laser, obs_orientation_to_goal, obs_distance_to_goal, obs_velocity, obs_timestep, rewards, actions, advantage=None, neglog=None):

        loss, pg_loss, value_loss, entropy, var = self._train([obs_laser, obs_orientation_to_goal, obs_distance_to_goal, obs_velocity, rewards, actions, advantage, neglog])


        return loss, pg_loss, value_loss, entropy, var


    def predict(self, obs_laser, obs_orientation_to_goal, obs_distance_to_goal, obs_velocity):
        action, values, neglogs = self._predict([obs_laser, obs_orientation_to_goal, obs_distance_to_goal, obs_velocity])
        return action, values, neglogs

    def saveWeights(self, path):
        self._model.save_weights(path + '.h5')

    def getWeights(self):
        return  self._model.get_weights()

    def setWeights(self, weights):
        self._model.set_weights(weights)

    def load_weights(self, path):
        self._model.load_weights(path)


    def policy_action_certain(self, s):
        """
        Use the actor to predict the next action to take, using the policy
        :param s: state of a single robot
        :return: [actions]
        """
        laser = np.array([np.array(s[i][0]) for i in range(0, len(s))]).swapaxes(0,1)
        orientation = np.array([np.array(s[i][1]) for i in range(0, len(s))]).swapaxes(0,1)
        distance = np.array([np.array(s[i][2]) for i in range(0, len(s))]).swapaxes(0,1)
        velocity = np.array([np.array(s[i][3]) for i in range(0, len(s))]).swapaxes(0,1)

        action = self._sample(
            [np.array([laser]), np.array([orientation]), np.array([distance]), np.array([velocity])])


        # These are the values of these two quantities, as Numpy arrays,
        # given our sample image of two elephants
        pooled_grads_value, conv_layer_output_value, output = self.iterate([np.array([laser]), np.array([orientation]), np.array([distance]), np.array([velocity])])

        # print("output: ", output)
        # print("Pooled grads value: ", pooled_grads_value)
        # print("Conv_Layer_output_value: ", conv_layer_output_value)
        # print("Output: ", output)
        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the elephant class
        for i in range(16):
            conv_layer_output_value[:, i] *= pooled_grads_value[i]

        # The channel-wise mean of the resulting feature map
        # is our heatmap of class activation
        # print("Conv Value 2: ", conv_layer_output_value)
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        # print(heatmap.shape)
        # print("Heatmap: ", heatmap)



        return (action, heatmap)




########################################################################
######################## Helper classes for NN #########################
########################################################################

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
        return concatenate([tmp, tmp * 0.0 + self._var], axis=-1)# I hate keras for this shit

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = 2
        return tuple(output_shape)




