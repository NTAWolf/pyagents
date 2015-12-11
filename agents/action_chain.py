from random import random

from . import Agent
from util.collections import CircularList
from util.listops import sublists, listhash
from util.interpolation import linear_latch


class ActionChainAgent(Agent):
    """docstring for RandomAgent"""

    def __init__(self, chain_length):
        super(ActionChainAgent, self).__init__(
            name='ActionChainAgent', version='1.2')
        self.q = dict()  # state-action values: q[state][action]
        self.chain = CircularList(chain_length)
        # e=1 until frame 5k, then interpolate down to e=0.05 in frame 10k,
        # and keep it there for the remaining time
        self.e_params = (5000, 10000, 1.0, 0.05)
        self.e = 0.5
        self.nframes = 0
        self.learning_rate = 0.1
        self.discount = 0.9
        self.last_action = None

    def update_e(self):
        self.e = linear_latch(self.nframes, *self.e_params)

    def select_action(self):
        # Always take random action first
        action = self.get_random_action()

        # Greedy action
        if random() > self.e and self.chain.full:
            res = self.get_greedy_action(self.available_actions)
            if res is not None:
                action = res

        self.chain.append(action)
        return action

    def receive_reward(self, reward):
        for chain in sublists(self.chain):
            # Consider the previous moves to be the current state
            state = chain[1:]
            action = chain[0]
            self.update_chain(state, action, reward)
        self.on_frame_end()

    def on_frame_end(self):
        self.nframes += 1
        self.update_e()

    def on_episode_start(self):
        pass

    def on_episode_end(self):
        pass

    def update_chain(self, state, action, reward):
        lhstate = listhash(state)
        if not lhstate in self.q:
            self.q[lhstate] = dict()
        if not action in self.q[lhstate]:
            self.q[lhstate][action] = reward
        else:
            val = self.q[lhstate][action]
            self.q[lhstate][action] = val + self.learning_rate * \
                (reward - self.discount * val)

    def get_greedy_action(self, available_actions):
        # Do a tree search in the previously seen states
        # that match the current state
        best_action = None
        best_value = None
        for state in sublists(self.chain):
            lhstate = listhash(state)
            if lhstate in self.q:
                s = self.q[lhstate]
                for a in available_actions:
                    if a in s:
                        val = s[a]
                        if val > best_value:
                            best_action = a
                            best_value = val
        return best_action

    def get_settings(self):
        settings = {'chain_length': self.chain.capacity(),
                    'e_params': self.e_params,
                    'learning_rate': self.learning_rate,
                    'discount': self.discount
                    }

        settings.update(super(ActionChainAgent, self).get_settings())

        return settings
