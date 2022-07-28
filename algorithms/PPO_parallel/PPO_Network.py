import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import datetime
import math
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, laser, dist_to_goal, ori_to_goal, velocity, action):
        self.laser = laser
        self.dist_to_goal = dist_to_goal
        self.ori_to_goal = ori_to_goal
        self.velocity = velocity
        self.action = action['action']
        self.action_neglog_policy = action['neglog_policy']
        self.action_advantage = action['advantage']
        self.action_reward = action['reward']
        self.length_dataset = len(laser)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.laser[idx], self.dist_to_goal[idx], self.ori_to_goal[idx], self.velocity[idx], self.action[idx], self.action_neglog_policy[idx], self.action_advantage[idx], self.action_reward[idx]


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        scan_size = 121
        self.lidar_conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3)
        in_f = self.get_in_features(h_in=scan_size, kernel_size=3)
        self.lidar_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        in_f = self.get_in_features(h_in=in_f, kernel_size=3)

        features_scan = int(in_f ** 2 * 32)

        self.lidar_flat = nn.Linear(in_features=features_scan, out_features=160)
        self.concated_some = nn.Linear(in_features=180, out_features=96)

        # Policy
        self.mu = nn.Linear(in_features=96, out_features=2)

        # Value
        self.value_temp = nn.Linear(out_features=128, in_features=96)
        self.value = nn.Linear(out_features=1, in_features=128, bias=False)

        print(self.summary())

    def forward(self, laser, orientation_to_goal, distance_to_goal, velocity):
        laser = F.relu(self.lidar_conv1(laser))
        laser = F.relu(self.lidar_conv2(laser))
        laser_flat = torch.flatten(laser)
        laser_flat = F.relu(self.lidar_flat(laser_flat))

        orientation_flat = torch.flatten(orientation_to_goal)
        distance_flat = torch.flatten(distance_to_goal)
        velocity_flat = torch.flatten(velocity)

        concat = torch.cat([orientation_flat, distance_flat, velocity_flat, laser_flat])
        densed = F.relu(self.concated_some(concat))

        mu = torch.tanh(self.mu(densed))
        var = torch.FloatTensor([0.0, 0.0])  # TODO:
        value = F.relu(self.value_temp(densed))
        value = F.relu(self.value(value))

        return [mu.to('cpu'), var.to('cpu'), value.to('cpu')]

    def get_in_features(self, h_in, padding=0, dilation=1, kernel_size=0, stride=1):
        return (((h_in + 2 * padding - dilation * (kernel_size - 1) -1) / stride) + 1)

    def summary(self):
        pass #summary(self, [(1, 4, 121, 121), (1, 2, 4), (1, 1, 4, 1), (1, 2, 4)])


class PPO_Network():
    NEEDED_OBSERVATIONS = ['lidar_0', 'orientation_to_goal', 'distance_to_goal', 'velocity']

    def __init__(self, act_dim, env_dim, args, load_weights=False):
        config = {
            'lidar_size': args.number_of_rays,
            'orientation_size': 2,
            'distance_size': 1,
            'velocity_size': 2,
            'stack_size': env_dim[0],
            'clipping_range': 0.2,
            'coefficient_value': 0.5,
            'learn_rate': 0.0001
        }
        self._name = type(self).__name__
        self._start_time = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M")
        self.args = args
        self.config = (config)
        self.loss_fn = nn.MSELoss()
        self.device = 'cpu'
        if args.use_cpu_only == "False":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Uploading model to {}".format(self.device))
        self._model = Model(self.config).to(self.device)
        print("Done!")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr=self.config["learn_rate"], momentum=0.9)
        #self.optimizer = torch.optim.Adam(self._model.parameters(), lr=self.config["learn_rate"])

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config

    def build(self):
        pass

    def _select_action_continuous_clip(self, mu, var):
        return torch.clamp(mu + torch.exp(var) * mu.normal_(0, 0.5), -1.0, 1.0)

    def _neglog_continuous(self, action, mu, var):
        return 0.5 * torch.sum(torch.square(action - mu) / torch.exp(var)) + 0.5 * math.log(2.0 * torch.pi) \
               * torch.FloatTensor([2.0]) + torch.sum(var)

    def entropy_continuous(selfself, var):
        return torch.sum(var + 0.5 * math.log(2.0 * torch.pi * math.e), axis=-1)

    def predict(self, obs_laser, obs_orientation_to_goal, obs_distance_to_goal, obs_velocity):
        '''
        observation: python dict with the keys:
        'laser_0', 'orientation_to_goal', 'distance_to_goal', 'velocity'.
        shape of each key: (num_agents, size_of_the_obs, stack_size).
        For the lidar with stack_size 4 and 2 agents: (2, 1081, 4)
        '''
        logging.info(f'Tracing predict function of {self.__class__}')
       obs_laser = torch.from_numpy(obs_laser).float()
        obs_laser = obs_laser.transpose(1, 3).to(self.device)
        obs_orientation_to_goal = torch.from_numpy(obs_orientation_to_goal).float().to(self.device)
        obs_distance_to_goal = torch.from_numpy(obs_distance_to_goal).float().to(self.device)
        obs_velocity = torch.from_numpy(obs_velocity).float().to(self.device)

        net_out = self._model.forward(obs_laser, obs_orientation_to_goal, obs_distance_to_goal, obs_velocity)

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
        return selected_action, neglog

    def train(self, observation, action):
        self._model.train()
        logging.info(f'Tracing train function of {self.__class__}')
        obs_laser = torch.from_numpy(observation['lidar_0']).float()
        obs_laser = obs_laser.transpose(1, 3)
        obs_orientation_to_goal = torch.from_numpy(observation['orientation_to_goal']).float()
        obs_distance_to_goal = torch.from_numpy(observation['distance_to_goal']).float()
        obs_velocity = torch.from_numpy(observation['velocity']).float()

        batch_size = 1
        worker = 0
        dataset = CustomDataset(obs_laser, obs_distance_to_goal, obs_orientation_to_goal, obs_velocity, action)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=worker, pin_memory=True)#, pin_memory_device=self.device)

        for laser, dist_to_goal, ori_to_goal, velocity, action, action_neglog_policy, action_advantage, action_reward in train_loader:
            self.optimizer.zero_grad()
            outputs = self._model.forward(laser.to(self.device), dist_to_goal.to(self.device), ori_to_goal.to(self.device), velocity.to(self.device))
            loss = self.calculate_loss(action, action_neglog_policy, action_advantage, action_reward, outputs)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self._model.eval()
        return {'loss': loss}

    def calculate_loss(self, action, action_neglog_policy, action_advantage, action_reward, net_out):
        neglogp = self._neglog_continuous(action, net_out[0], net_out[1])

        ratio = torch.exp(torch.FloatTensor([action_neglog_policy]) - neglogp)
        pg_loss = -torch.FloatTensor([action_advantage]) * ratio
        pg_loss_cliped = -torch.FloatTensor([action_advantage]) * torch.clamp(ratio, 1.0 - self._config[
            'clipping_range'], 1.0 + self._config['clipping_range'])

        pg_loss = torch.mean(torch.max(pg_loss, pg_loss_cliped))

        value_loss = self.loss_fn(net_out[2], torch.FloatTensor([action_reward])) * self._config[
            'coefficient_value']

        loss = pg_loss + value_loss - self.entropy_continuous(net_out[1])

        return loss

    def load_weights(self, path):
        self._model.load_weights(path)

    def load_model(self, path):
        # path = path.replace('\\', '/')
        print(path)
        # self._model = tf.keras.models.load_model(path)
        self.print_summary()

    def pedict_certain(self, s):
        """
        Use the actor to predict the next action to take, using the policy
        :param s: state of a single robot
        :return: [actions]
        """
        #laser = np.array([np.array(s[i][0]) for i in range(0, len(s))]).swapaxes(0, 1)
        laser = np.array([np.array(s[i][0]) for i in range(0, len(s))]).swapaxes(0,2)
        #print("laser_state2: ", laser.shape)
        orientation = np.array([np.array(s[i][1]) for i in range(0, len(s))]).swapaxes(0, 1)
        distance = np.array([np.array(s[i][2]) for i in range(0, len(s))]).swapaxes(0, 1)
        velocity = np.array([np.array(s[i][3]) for i in range(0, len(s))]).swapaxes(0, 1)

        if self.args.lidar_activation:
            return self.make_gradcam_heatmap(laser, orientation, distance, velocity, 1)
        else:
            net_out = self._model([np.expand_dims(laser, 0), np.expand_dims(orientation, 0), np.expand_dims(distance, 0),
                                   np.expand_dims(velocity, 0)])

            return net_out[0], None

    def print_summary(self):
        self._model.summary()

    def set_model_weights(self, weights):
        self._model.load_state_dict(weights)

    def get_model_weights(self):
        return self._model.state_dict()

    def save_model_weights(self, path):
        torch.save(self._model, path + ".pt")

    def make_gradcam_heatmap(self, laser, orientation, distance, velocity, pred_index=0):
        """

        :param laser:
        :param orientation:
        :param distance:
        :param velocity:
        :param pred_index: 0 for activations of linVel 1 for activations of angular velocity
        :return:
        """

        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [self._model.inputs], [self._model.get_layer('body_lidar-conv_2').output, self._model.output[0]])

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model([np.expand_dims(laser, 0), np.expand_dims(orientation, 0), np.expand_dims(distance, 0), np.expand_dims(velocity, 0)])
            class_channel = preds[:, pred_index]


        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)


        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))


        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return (preds, heatmap.numpy())

    def create_perception_model(self):
        layer_name = 'body_lidar-dense'
        proximity_predictions = Dense(3, activation='softmax')(self._model.get_layer(layer_name).output)
        self._perception_model = keras.Model([self._model.inputs],[proximity_predictions])
        self._perception_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        print(self._perception_model.summary())

    def make_proximity_prediction(self, s):
        laser = np.array([np.array(s[i][0]) for i in range(0, len(s))]).swapaxes(0, 1)
        orientation = np.array([np.array(s[i][1]) for i in range(0, len(s))]).swapaxes(0, 1)
        distance = np.array([np.array(s[i][2]) for i in range(0, len(s))]).swapaxes(0, 1)
        velocity = np.array([np.array(s[i][3]) for i in range(0, len(s))]).swapaxes(0, 1)

        proximity_categories = self._perception_model([np.expand_dims(laser, axis=0), np.expand_dims(orientation, axis=0), np.expand_dims(distance, axis=0), np.expand_dims(velocity, axis=0)])
        #
        # proximity_categories = proximityFunc(tf.convert_to_tensor(np.expand_dims(laser, axis=0), dtype='float64'),
        #                                     tf.convert_to_tensor(np.expand_dims(orientation, axis=0), dtype='float64'),
        #                                     tf.convert_to_tensor(np.expand_dims(distance, axis=0), dtype='float64'),
        #                                     tf.convert_to_tensor(np.expand_dims(velocity, axis=0), dtype='float64'))
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
                # inputs = np.append(inputs, np.array([laser, orientation, distance, velocity]))
                inputsL = np.append(inputsL, np.expand_dims(laser, axis=0), axis=0)
                inputsO = np.append(inputsO, np.expand_dims(orientation, axis=0), axis=0)
                inputsD = np.append(inputsD, np.expand_dims(distance, axis=0), axis=0)
                inputsV = np.append(inputsV, np.expand_dims(velocity, axis=0), axis=0)

        proximity_categories = np.asarray(proximity_categories)
        self._perception_model.fit([inputsL, inputsO, inputsD, inputsV], proximity_categories, shuffle=True)

