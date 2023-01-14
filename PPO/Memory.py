class Memory:   # collected from old policy
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.reached_goal = []
        self.logprobs = []

    def clear_memory(self):
        del self.observations[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.reached_goal[:]
        del self.logprobs[:]

    def __len__(self):
        return len(self.observations)