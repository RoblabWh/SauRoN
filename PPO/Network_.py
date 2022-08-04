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
#from DebugListener import DebugListener

class CustomDataset(Dataset):
    def __init__(self, laser, dist_to_goal, ori_to_goal, velocity, action):
        self.laser = laser
        self.dist_to_goal = dist_to_goal
        self.ori_to_goal = ori_to_goal
        self.velocity = velocity
        self.action = torch.from_numpy(action['action']).float()
        self.action_neglog_policy = torch.from_numpy(action['neglog_policy']).float()
        self.action_advantage = torch.from_numpy(action['advantage']).float()
        self.action_reward = torch.from_numpy(action['reward']).float()
        self.length_dataset = len(self.laser)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.laser[idx], self.dist_to_goal[idx], self.ori_to_goal[idx], \
               self.velocity[idx], self.action[idx], self.action_neglog_policy[idx], \
               self.action_advantage[idx], self.action_reward[idx]


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        scan_size = 121
        self.lidar_conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=16, stride=8)
        in_f = self.get_in_features(h_in=scan_size, kernel_size=16, stride=8)
        self.lidar_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=8, stride=4)
        in_f = self.get_in_features(h_in=in_f, kernel_size=8, stride=4)

        features_scan = int((in_f**2) * 32)
        self.flatten = nn.Flatten()
        self.lidar_flat = nn.Linear(in_features=128, out_features=160)
        self.concated_some = nn.Linear(in_features=180, out_features=96)

        # Policy
        self.mu = nn.Linear(in_features=96, out_features=2)

        # Value
        self.value_temp = nn.Linear(out_features=128, in_features=96)
        self.value_temp2 = nn.Linear(out_features=128, in_features=96)
        self.value = nn.Linear(out_features=1, in_features=256, bias=False)

        # Var
        self.dense_var = nn.Linear(in_features=96, out_features=2)
        self.softplus = torch.nn.Softplus()

        print(self.summary())

    def forward(self, laser, orientation_to_goal, distance_to_goal, velocity):
        laser = F.relu(self.lidar_conv1(laser))
        laser = F.relu(self.lidar_conv2(laser))
        laser_flat = self.flatten(laser)

        laser_flat = F.relu(self.lidar_flat(laser_flat))

        orientation_flat = self.flatten(orientation_to_goal)
        distance_flat = self.flatten(distance_to_goal)
        velocity_flat = self.flatten(velocity)
        concat = torch.cat([laser_flat, orientation_flat, distance_flat, velocity_flat], dim=1)
        densed = F.relu(self.concated_some(concat))

        mu = torch.tanh(self.mu(densed))
        var = self.softplus(self.dense_var(densed)) #torch.FloatTensor([0.0, 0.0])  # TODO:
        value1 = F.relu(self.value_temp(densed))
        value2 = F.relu(self.value_temp2(densed))
        value_cat = torch.cat([value1, value2], dim=1)
        value = F.relu(self.value(value_cat))

        return [mu.to('cpu'), var.to('cpu'), value.to('cpu')]

    def get_in_features(self, h_in, padding=0, dilation=1, kernel_size=0, stride=1):
        return (((h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)

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
        self._model.train()
        print("Done!")
        self.criterion = nn.CrossEntropyLoss()
        #self.debugListener = DebugListener()
        #self.optimizer = torch.optim.SGD(self._model.parameters(), lr=self.config["learn_rate"], momentum=0.9)
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=self.config["learn_rate"])

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config

    def build(self):
        pass

    def _select_action_continuous_clip(self, mu, sigma):
        #self.debugListener.debug2(sigma)
        return torch.clamp(torch.normal(mu, sigma), -1.0, 1.0)
        #return torch.clamp(mu + torch.exp(var) * mu.normal_(0, 0.5), -1.0, 1.0)

    def _neglog_continuous(self, action, mu, sigma):
        variance = torch.square(sigma)
        pdf = 1. / torch.sqrt(2. * np.pi * variance) * torch.exp(
            -torch.square(action - mu) / (2. * variance))
        pdf = torch.sum(pdf)
        log_pdf = torch.log(pdf + 1e-10)
        return log_pdf

        # return 0.5 * torch.sum(torch.square(action - mu) / torch.exp(var)) + 0.5 * math.log(2.0 * torch.pi) \
        #        * torch.FloatTensor([2.0]) + torch.sum(var)

    def entropy_continuous(selfself, sigma):
        #loss_entropy = torch.sum(var + 0.5 * math.log(2.0 * torch.pi * math.e), axis=-1)
        loss_entropy = 0.0001 * torch.mean(- (torch.log(2 * np.pi * torch.square(sigma)) + 1) / 2)
        return loss_entropy

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
        obs_orientation_to_goal = torch.from_numpy(obs_orientation_to_goal).transpose(1, 2).float().to(self.device)
        obs_distance_to_goal = torch.from_numpy(obs_distance_to_goal).float()[0].to(self.device)
        obs_velocity = torch.from_numpy(obs_velocity).transpose(1, 2).float().to(self.device)

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

        batch_size = len(obs_laser)
        worker = 0
        dataset = CustomDataset(obs_laser, obs_distance_to_goal, obs_orientation_to_goal, obs_velocity, action)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=worker, pin_memory=True)#, pin_memory_device=self.device)

        for laser, dist_to_goal, ori_to_goal, velocity, action, action_neglog_policy, action_advantage, action_reward in train_loader:
            laser = laser.to(self.device)
            dist_to_goal = dist_to_goal.to(self.device)
            ori_to_goal = ori_to_goal.to(self.device)
            velocity = velocity.to(self.device)
            outputs = self._model.forward(laser, dist_to_goal, ori_to_goal, velocity)
            loss = self.calculate_loss(action, action_neglog_policy, action_advantage, action_reward, outputs)
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self._model.eval()
        return {'loss': loss}

    def calculate_loss(self, action, action_neglog_policy, action_advantage, action_reward, net_out):
        neglogp = self._neglog_continuous(action, net_out[0], net_out[1])
        #neglogp = self._neglog_continuous(action[i], net_out[0][i], torch.FloatTensor([0.0, 0.0]))

        ratio = torch.exp(action_neglog_policy - neglogp)
        pg_loss = -action_advantage * ratio
        pg_loss_cliped = -action_advantage * torch.clamp(ratio, 1.0 - self._config[
            'clipping_range'], 1.0 + self._config['clipping_range'])

        pg_loss = torch.mean(torch.max(pg_loss, pg_loss_cliped))

        value_loss = self.loss_fn(net_out[2].squeeze(), action_reward) * self._config[
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
            laser = torch.from_numpy(np.expand_dims(laser, 0)).transpose(1, 3).float().to(self.device)
            orientation = torch.from_numpy(np.expand_dims(orientation, 0)).transpose(1, 2).float().to(self.device)
            distance = torch.from_numpy(np.expand_dims(distance, 0)).transpose(1, 2).float().to(self.device)
            velocity = torch.from_numpy(np.expand_dims(velocity, 0)).transpose(1, 2).float().to(self.device)
            net_out = self._model(laser, orientation, distance, velocity)

            return net_out[0], None

    def print_summary(self):
        self._model.summary()

    def set_model_weights(self, weights):
        self._model.load_state_dict(weights)

    def get_model_weights(self):
        return self._model.state_dict()

    def save_model_weights(self, path):
        torch.save(self._model, path + ".pt")