class CMD():
    def __init__(args, act_dim, env_dim, loadWeightsPath):
        self.args = args
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.loadWeightsPath = loadWeightsPath
        self.model = PPO_Multi(self.app, act_dim, env_dim, args)
        self.currentEpisode = 0
        self.progressbarWidget = Progressbar(self.currentEpisode, self.args)
        self.visibilities = 0 #TODO
        self.done = False

    def train(self):
        #TODO:update worker data
        self.done, levelNames = self.model.prepare_training(self.loadWeightsPath)

        while self.done == False:
            episodeDoneFuture = self.model.train_with_feedback_for_n_steps(self.visibilities)
            self.done, avrgRewardLastEpisode, successrates, currentEpisode, successAll = self.model.train_with_feedback_end_of_episode()
            self.currentEpisode = currentEpisode
            self.progressbarWidget.updateProgressbar(currentEpisode)
            self.tableWidget.updateAvrgRewardLastEpisode(avrgRewardLastEpisode)
            self.tableWidget.updateSuccessrate(successrates)
            self.successLabel.setText("Success insgesamt: " + str(successAll))