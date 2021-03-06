from . import Agent


class UniformRandomAgent(Agent):
    """Selects uniformly randomly among the available actions"""

    def __init__(self):
        super(UniformRandomAgent, self).__init__(
            name='UniformRandom', version='1')

    def select_action(self):
        return self.get_random_action()

    def receive_reward(self, reward):
        pass

    def on_episode_start(self):
        pass

    def on_episode_end(self):
        pass

    def reset(self):
        pass

