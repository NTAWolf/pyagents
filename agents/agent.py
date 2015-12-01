from random import randrange
import numpy as np


class Agent(object):
    """This class defines the agent interface to be used in this project.
    """

    def __init__(self, name, version):
        self.name = name
        self.version = version

    def select_action(self, state):
        """Returns one of the actions set in set_available_actions.
        """
        raise NotImplementedError("Method not implemented")

    def receive_reward(self, reward):
        raise NotImplementedError("Method not implemented")

    def get_random_action(self):
        return self.available_actions[randrange(len(self.available_actions))]

    def set_available_actions(self, actions):
        self.available_actions = list(actions)

    def on_episode_start(self):
        """Called on episode start by the GameManager
        """
        raise NotImplementedError("Method not implemented")

    def on_episode_end(self):
        """Called on episode end by the GameManager
        """
        raise NotImplementedError("Method not implemented")

    def get_settings(self):
        """Called by the GameManager when it is
        time to store this agent's settings

        Returns a dict
        """
        return dict([("name", self.name),
                     ("version", self.version),
                     ])
