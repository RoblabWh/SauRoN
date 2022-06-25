
from algorithms.PPO_parallel.PPO_Multi import PPO_Multi

class CMD():
    def __init__(self, args, act_dim, env_dim, loadWeightsPath = ""):
        self.args = args
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.loadWeightsPath = loadWeightsPath
        self.model = PPO_Multi(None, self.act_dim, self.env_dim, args)
        self.currentEpisode = 0
        self.visibilities = [False for _ in range(args.parallel_envs)]
        self.visibilities[0] = True
        self.done = False

    def train(self):
        self.done, levelNames = self.model.prepare_training(self.loadWeightsPath)
        while self.done == False:
            if self.model.train_with_feedback_for_n_steps(self.visibilities):
                self.done, avrgRewardLastEpisode, successrates, currentEpisode, successAll = self.model.train_with_feedback_end_of_episode()
                self.currentEpisode = currentEpisode
