class Memory:   # collected from old policy
    """
    Memory class used to store the transitions that the agent observes, and
    on which it will be trained.
    """
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