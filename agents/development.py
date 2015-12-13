import numpy as np

from . import Agent

class DevelopmentAgent(Agent):
    """This agent is for development purposes.
    It is continuosly edited.
    """

    def __init__(self):
        super(DevelopmentAgent, self).__init__(
            name='DevelopmentAgent', version='2')
        self.c = 0

    def select_action(self):
        if self.c > 100:
            raw = self.raw_state_callbacks.raw()
            print raw.shape
            # np.save('verytemp.npy', raw)
            # import sys
            # sys.exit()
        self.c += 1

        return super(DevelopmentAgent, self).get_random_action()

    def receive_reward(self, reward):
        pass

    def on_episode_start(self):
        print "Agent sees that episode starts"

    def on_episode_end(self):
        print "Agent sees that episode ends"

    def reset(self):
        pass

    def get_settings(self):
        return super(DevelopmentAgent, self).get_settings()
