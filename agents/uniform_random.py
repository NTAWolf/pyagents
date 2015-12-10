from . import Agent
from random import randrange


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

