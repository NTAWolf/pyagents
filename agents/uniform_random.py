from pyagents.basis import Agent
from random import randrange

class UniformRandomAgent(Agent):
    """docstring for RandomAgent"""
    def __init__(self):
        super(RandomAgent, self).__init__(name='UniformRandom', version='1')

    def select_action(self, state, available_actions):
        """Returns one of the actions given in available_actions.
        """
        return available_actions[randrange(len(available_actions))]

    def receive_reward(self, reward):
        pass
        
