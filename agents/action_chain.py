from random import randrange, random

from . import Agent
from util.collections import CircularList
from util.listops import sublists, listhash


class ActionChainAgent(Agent):
    """docstring for RandomAgent"""

    def __init__(self, chain_length):
        super(ActionChainAgent, self).__init__(
            name='ActionChainAgent', version='1')
        self.q = dict()  # state-action values: q[state][action]
        self.chain = CircularList(chain_length)
        self.e = 0.25
        self.learning_rate = 0.1
        self.discount = 0.9
        self.last_action = None

    def select_action(self, state, available_actions):
        """Returns one of the actions given in available_actions.
        """
        # Always take random action first
        action = self.get_random_action(available_actions)
        
        # Greedy action
        if random() > self.e and self.chain.full:
            res = self.get_greedy_action(available_actions)
            if res is not None:
                action = res

        self.chain.insert(action)
        return action

    def receive_reward(self, reward):
        for chain in sublists(self.chain):
            state = chain[1:] # Consider the previous moves to be the current state
            action = chain[0]
            self.update_chain(state, action, reward)

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
            self.q[lhstate][action] = val + self.learning_rate*(reward - self.discount * val)

    def get_random_action(self, available_actions):
        return available_actions[randrange(len(available_actions))]

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

