import numpy as np

from . import Agent
from util.pongcess import StateIndex, RelativeBall, Positions,RelativeIntercept

import matplotlib.pyplot as plt
import os

def save_image(data, file_path):
    path = os.path.join('fig', file_path)
    aximg = plt.imshow(data)
    plt.savefig(path)
    print "Saved figure in {}".format(path)

startshow = 60
endshow = 300
showfreq = 1

actionmap = {0:'noop', 3:'up', 4:'down'}

class SmartAgent(Agent):
    """
    Agent a SARSA(lambda) approach
    Input RGB image is preprocessed, resulting in states
    - (x, y) ball
    - y player
    - y opponent
    """

    def __init__(self, n_frames_per_action=4):
        super(SmartAgent, self).__init__(name='Smart', version='1')
        self.n_frames_per_action = n_frames_per_action
        self.last_action = 0
        self.n = 0
        
    def select_action(self):

        self.n += 1
        if self.n < startshow:# or self.n % showfreq != 0:
            return self.last_action

        p = self.preprocessor
        res = p.process()

        if res == 0:
            action = 0
        elif res == 1:
            action = 3
        else: # res == -1
            action = 4

        # i = self.preprocessor.intercept
        # frame = self.rgb()
        # frame[i[1],i[0],:] = 255
        # save_image(frame, 'frame_{}_p1_{}_ag_{}.png'.format(self.n, p.p1, p.ag))

        # if self.n > endshow:
        #     import sys
        #     sys.exit()

        self.last_action = action
        return action

    def set_available_actions(self, actions):
        super(SmartAgent, self).set_available_actions(actions)

    def set_raw_state_callbacks(self, state_functions):
        self.preprocessor = RelativeIntercept(state_functions)
        self.rgb = state_functions.rgb

    def receive_reward(self, reward):
        pass

    def reset(self):
        pass

    def get_settings(self):
        settings =  {
            "name": self.name,
            "version": self.version,
            "preprocessor": self.preprocessor.get_settings(),
        }

        settings.update(super(SmartAgent, self).get_settings())
        
        return settings
