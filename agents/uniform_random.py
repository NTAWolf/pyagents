from . import Agent
from random import randrange


class UniformRandomAgent(Agent):
    """docstring for RandomAgent"""

    def __init__(self):
        super(UniformRandomAgent, self).__init__(
            name='UniformRandom', version='1')

    def select_action(self, state, available_actions):
        """Returns one of the actions given in available_actions.
        """
        return self.get_random_action(available_actions)

    def receive_reward(self, reward):
        pass

    def on_episode_start(self):
        pass

    def on_episode_end(self):
        pass

    def get_settings(self):
        """Called by the GameManager when it is
        time to store this object's settings

        Returns a dict representing the settings needed to 
        reproduce this object.
        """
        return dict()

