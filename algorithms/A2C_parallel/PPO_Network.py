import numpy as np
import tensorflow.keras as k

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, concatenate, MaxPool1D, Lambda, Layer, InputSpec

from tensorflow import maximum, reduce_mean, exp, function, squeeze, reduce_sum, square, shape, cast, clip_by_value, gradients  # TODO Prüfen ob clip_by_norm richtig ist (und gradients und function)
import tensorflow as tf
from tensorflow.keras.models import Model


class PPO_Network:
    """
    Neural Network for Actor Critic
    """

    def __init__(self, act_dim, env_dim, args):
        """ Initialization
        """
        #print(k.__version__)
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
        #self._input_timestep = Input(shape=(1,env_dim[0],), dtype='float32', name='input_Timestep')

        tf.compat.v1.enable_eager_execution()

        tf.config.experimental_run_functions_eagerly(True)

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

        fully_connect = self.buildMainNet("shared" if self._shared else "policy", self._network_size)

        if self._shared:
            fully_connect2 = fully_connect
        else:
            fully_connect2 = self.buildMainNet("value", self._network_size)

        # Policy
        # mu = Dense(units=2, activation='tanh', name='output_mu')(fully_connect)
        # var = ContinuousLayer(name='output_continous')(mu)

        # # actor
        self._mu_var = ContinuousLayer()(fully_connect)
        mu = Lambda(lambda x: x[:, :2])(self._mu_var)
        var = Lambda(lambda x: x[:, -2:])(self._mu_var)

        # Value
        if self.args.load_christian:
            value = Dense(units=1, activation='linear', name='out_value')(fully_connect2)
        else:
            value = Dense(units=128, activation='relu', name='out_value_dense')(fully_connect2)
            value = Dense(units=1, activation=None, use_bias=False, name='out_value')(value)

        # Create the Keras Model
        self._model = Model(inputs=[self._input_laser, self._input_orientation, self._input_distance, self._input_velocity],
                            outputs=[mu, var, value])

        # Create the Optimizer
        self._optimizer = Adam(learning_rate=self.lr, epsilon=1e-5, clipnorm=1.0)

        #self.printSummary()

    def predict(self, obs_laser, obs_orientation_to_goal, obs_distance_to_goal, obs_velocity):
        a = tf.function(self.predict_function)
        return a(tf.convert_to_tensor(obs_laser, dtype='float64'), tf.convert_to_tensor(obs_orientation_to_goal, dtype='float64'), tf.convert_to_tensor(obs_distance_to_goal, dtype='float64'),tf.convert_to_tensor(obs_velocity, dtype='float64'))


    # TODO Laser Strahlen Anzahl aus args holen
    @tf.function(input_signature=[tf.TensorSpec((None, 1081, 4), dtype='float64'),
                                   tf.TensorSpec((None, 2, 4), dtype='float64'),
                                   tf.TensorSpec((None, 1, 4), dtype='float64'),
                                   tf.TensorSpec((None, 2, 4), dtype='float64')])
    def predict_function(self, obs_laser, obs_orientation_to_goal, obs_distance_to_goal, obs_velocity):
        '''
        observation: python dict with the keys:
        'laser_0', 'orientation_to_goal', 'distance_to_goal', 'velocity'.
        shape of each key: (num_agents, size_of_the_obs, stack_size).
        For the lidar with stack_size 4 and 2 agents: (2, 1081, 4)
        '''
        net_out = self._model([obs_laser, obs_orientation_to_goal, obs_distance_to_goal, obs_velocity]) #TODO observation vernuenftig an model übergeben

        selected_action, neglog = self._postprocess_predictions(*net_out)

        return [selected_action, net_out[2], neglog]

    def _postprocess_predictions(self, mu, var, val):
        """
        Calculates the action selection and the neglog based on the network output mu, var, value.

        Parameters:
            mu (Tensor (None, 2)): The mu output from the dnn.
            var (Tensor (None, 2)): The var output from the dnn.
            val (Tensor (None, 1)): The value output from the dnn.

        Returns:
            selected_action (Tensor (None, 2))
            neglog (Tensor (None,))

        """
        selected_action = self._select_action_continuous_clip(mu, var)
        neglog = self._neglog_continuous(selected_action, mu, var)
        return (selected_action, neglog)


    def _select_action_continuous_clip(self, mu, var):
        return clip_by_value(mu + exp(var) * tf.random.normal(shape(mu)), -1.0, 1.0)

    def _neglog_continuous(self, action, mu, var):
        return 0.5 * reduce_sum(square((action - mu) / exp(var)), axis=-1) \
                + 0.5 * tf.math.log(2.0 * np.pi) * cast(shape(action)[-1], dtype='float32') \
                + reduce_sum(var, axis=-1)

    def train_net(self, obs_laser, obs_orientation_to_goal, obs_distance_to_goal, obs_velocity, obs_timestep, rewards,
                  actions, advantage=None, neglog=None, values=None):
        func = tf.function(self.train_function)
        func({'laser_0': obs_laser, 'orientation_to_goal': obs_orientation_to_goal, 'distance_to_goal': obs_distance_to_goal, 'velocity': obs_velocity},
             {'action': actions,  'value': values, 'neglog_policy': neglog, 'reward': rewards,  'advantage': advantage})


    # TODO Laser Strahlen Anzahl aus args holen
    @tf.function(input_signature=[{'laser_0': tf.TensorSpec((None, 1081, 4), dtype='float64'),
                                   'orientation_to_goal': tf.TensorSpec((None, 2, 4), dtype='float64'),
                                   'distance_to_goal': tf.TensorSpec((None, 1, 4), dtype='float64'),
                                   'velocity': tf.TensorSpec((None, 2, 4), dtype='float64')},
                                  {'action': tf.TensorSpec((None, 2), dtype='float64'),
                                   'value': tf.TensorSpec((None), dtype='float64'),
                                   'neglog_policy': tf.TensorSpec((None), dtype='float64'),
                                   'reward': tf.TensorSpec((None), dtype='float64'),
                                   'advantage': tf.TensorSpec((None), dtype='float64')}])
    def train_function(self, observation, action):
        with tf.GradientTape() as tape:
            net_out = self._model(observation.values())
            loss = self.calculate_loss(observation, action, net_out)

        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

    def calculate_loss(self, observation, action, net_out):

        action['action'] = tf.cast(action['action'], tf.float32)
        neglogp = self._neglog_continuous(action['action'], net_out[0], net_out[1])

        action['neglog_policy'] = tf.cast(action['neglog_policy'], tf.float32)
        ratio = tf.exp(action['neglog_policy'] - neglogp)

        action['advantage'] = tf.cast(action['advantage'], tf.float32)
        pg_loss = -action['advantage'] * ratio
        pg_loss_cliped = -action['advantage'] * tf.clip_by_value(ratio, 1.0 - self._clip_range,
                                                                 1.0 + self._clip_range)

        pg_loss = tf.reduce_mean(tf.maximum(pg_loss, pg_loss_cliped))

        value_loss = mean_squared_error(tf.squeeze(net_out[2]), action['reward']) * self._coefficient_value

        value_loss = tf.cast(value_loss, tf.float32)
        loss = pg_loss + value_loss  # TODO Was ist mit Entropie? Warum kann die weg? Christian

        # entropy
        # entropy = self.entropy_continuous(self._var)
        # entropy = reduce_mean(entropy) * self._coefficient_entropy
        #
        # loss = pg_loss + value_loss - entropy

        return loss

    @tf.function(input_signature=[tf.TensorSpec((None, 1081, 4), dtype='float64'),
                                   tf.TensorSpec((None, 2, 4), dtype='float64'),
                                   tf.TensorSpec((None, 1, 4), dtype='float64'),
                                   tf.TensorSpec((None, 2, 4), dtype='float64')])
    def predict_certain(self,  obs_laser, obs_orientation_to_goal, obs_distance_to_goal, obs_velocity):
        '''
        observation: python dict with the keys:
        'laser_0', 'orientation_to_goal', 'distance_to_goal', 'velocity'.
        shape of each key: (num_agents, size_of_the_obs, stack_size).
        For the lidar with stack_size 4 and 2 agents: (2, 1081, 4)
        '''
        last_conv_layer_output = self._model.get_layer("shared" if self._shared else "policy" + '_conv1d_laser_last').output
        grad_model = tf.keras.models.Model(
            [self._model.inputs], [last_conv_layer_output, self._model.output[0]]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model([obs_laser, obs_orientation_to_goal, obs_distance_to_goal, obs_velocity])
            class_channel = preds[:, 0] # 0 should tell the gradient tape to watch the lin vel (1 angular) ... at least we hope so...

        # This is the gradient of the output neuron
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        # pooled_grads = mean(grads, axis=(0, 1))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        return [preds, heatmap.numpy()]



    @tf.function(input_signature=[tf.TensorSpec((None, 1081, 4), dtype='float64'),
                                  tf.TensorSpec((None, 2, 4), dtype='float64'),
                                  tf.TensorSpec((None, 1, 4), dtype='float64'),
                                  tf.TensorSpec((None, 2, 4), dtype='float64')])
    def predict_proximity(self, obs_laser, obs_orientation_to_goal, obs_distance_to_goal, obs_velocity):
        '''
        observation: python dict with the keys:
        'laser_0', 'orientation_to_goal', 'distance_to_goal', 'velocity'.
        shape of each key: (num_agents, size_of_the_obs, stack_size).
        For the lidar with stack_size 4 and 2 agents: (2, 1081, 4)
        '''
        preds = self._perception_model([obs_laser, obs_orientation_to_goal, obs_distance_to_goal, obs_velocity])
        return preds

    def entropy_continuous(self, var):
        return reduce_sum(var + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def printSummary(self):
        self._model.summary()

    def saveWeights(self, path):
        self._model.save_weights(path + '.h5')

    def getWeights(self):
        return  self._model.get_weights()

    def setWeights(self, weights):
        self._model.set_weights(weights)

    def load_weights(self, path):
        self._model.load_weights(path)


    def create_perception_model(self):
        layer_name = 'policy_dense_laser'  # TODO schauen ob es wirklich shared ist
        proximity_predictions = Dense(3, activation='softmax')(self._model.get_layer(layer_name).output)
        self._perception_model = Model(
            inputs=[self._input_laser, self._input_orientation, self._input_distance, self._input_velocity],
            outputs=proximity_predictions)
        self._perception_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        print(self._perception_model.summary())

    def make_proximity_prediction(self, s):
        laser = np.array([np.array(s[i][0]) for i in range(0, len(s))]).swapaxes(0, 1)
        orientation = np.array([np.array(s[i][1]) for i in range(0, len(s))]).swapaxes(0, 1)
        distance = np.array([np.array(s[i][2]) for i in range(0, len(s))]).swapaxes(0, 1)
        velocity = np.array([np.array(s[i][3]) for i in range(0, len(s))]).swapaxes(0, 1)

        proximityFunc = tf.function(self.predict_proximity)

        proximity_categories = proximityFunc(tf.convert_to_tensor(np.expand_dims(laser, axis=0), dtype='float64'),
                                            tf.convert_to_tensor(np.expand_dims(orientation, axis=0), dtype='float64'),
                                            tf.convert_to_tensor(np.expand_dims(distance, axis=0), dtype='float64'),
                                            tf.convert_to_tensor(np.expand_dims(velocity, axis=0), dtype='float64'))
        return proximity_categories

    def train_perception(self, states, proximity_categories):
        inputsL = np.array([])
        inputsO = np.array([])
        inputsD = np.array([])
        inputsV = np.array([])
        inputs = np.array([])
        for i, s in enumerate(states):
            laser = np.array([np.array(s[i][0]).astype('float32') for i in range(0, len(s))]).swapaxes(0, 1)
            orientation = np.array([np.array(s[i][1]).astype('float32') for i in range(0, len(s))]).swapaxes(0, 1)
            distance = np.array([np.array(s[i][2]).astype('float32') for i in range(0, len(s))]).swapaxes(0, 1)
            velocity = np.array([np.array(s[i][3]).astype('float32') for i in range(0, len(s))]).swapaxes(0, 1)

            if i == 0:
                inputsL = np.array([laser])
                inputsO = np.array([orientation])
                inputsD = np.array([distance])
                inputsV = np.array([velocity])
            else:
                #inputs = np.append(inputs, np.array([laser, orientation, distance, velocity]))
                inputsL = np.append(inputsL,np.expand_dims(laser, axis=0), axis=0)
                inputsO = np.append(inputsO, np.expand_dims(orientation, axis=0), axis=0)
                inputsD = np.append(inputsD, np.expand_dims(distance, axis=0), axis=0)
                inputsV = np.append(inputsV, np.expand_dims(velocity, axis=0), axis=0)
        proximity_categories = np.asarray(proximity_categories)#.astype('float64')
        print(inputsL.shape, inputsO.shape, inputsD.shape, inputsV.shape, proximity_categories.shape)
        print(proximity_categories)

        # tensorL = tf.convert_to_tensor(inputsL, dtype=tf.float32)
        # tensorO = tf.convert_to_tensor(inputsO, dtype=tf.float32)
        # tensorD = tf.convert_to_tensor(inputsD, dtype=tf.float32)
        # tensorV = tf.convert_to_tensor(inputsV, dtype=tf.float32)



        self._perception_model.fit([inputsL, inputsO, inputsD, inputsV], proximity_categories)


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

        actionCertainFunc = tf.function(self.predict_certain)

        action, heatmap = actionCertainFunc(tf.convert_to_tensor(np.expand_dims(laser, axis=0), dtype='float64'), tf.convert_to_tensor(np.expand_dims(orientation, axis=0), dtype='float64'),
                                   tf.convert_to_tensor(np.expand_dims(distance, axis=0), dtype='float64'),tf.convert_to_tensor(np.expand_dims(velocity, axis=0), dtype='float64'))

        return (action, heatmap)


########################################################################
######################## Helper classes for NN #########################
########################################################################
import tensorflow.keras as K

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