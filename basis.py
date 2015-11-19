# agent.py

import util
from datetime import datetime

class Agent(object):
    """This class defines the agent interface to be used in this project.
    """

    def __init__(self, name):
        self.name = name

    def select_action(self, state, available_actions):
        """Returns one of the actions given in available_actions.
        """
        raise NotImplementedError("Method not implemented")

    def receive_reward(self, reward):
        raise NotImplementedError("Method not implemented")


class GameManager(object):
    """This class takes care of the interactions between an agent and
    a game across episodes, as well as overall logging of performance.
    """

    def __init__(self, game_name, agent, results_dir):
        self.game_name = game_name
        self.agent = agent
        self.ale = ALEInterface()
        self.ale.loadROM(game_name)

        now = datetime.now().strftime('%Y%m%d-%H-%M')
        self.results_dir = os.join(results_dir, game_name[:-4] + now) # drop .bin, append current time down to the minute
        self.initiate_results_dir()

        self.log = util.Logger(('settings','action', 'episode','run'), 
                                'episode', os.join(self.results_dir, 'GameManager.log'))


    def initiate_results_dir(self):
        os.makedirs(self.results_dir) # Should raise an error if directory exists

    def run(self, n_episodes):
        self.log.run("Starting run for {} episodes".format(n_episodes))
        start = datetime.now()
        for episode in xrange(10):
            self.log.episode("Starting episode {}".format(episode))
            self._run_episode()
        duration = datetime.now() - start
        self.log.run("Finished run after {}".format(duration))

    def _run_episode(self):
        start = datetime.now()
        total_reward = 0
        while not self.ale.game_over():
            action = self.agent.select_action(state, legal_actions)
            reward = self.ale.act(action)
            self.agent.receive_reward()
            total_reward += reward
        duration = datetime.now() - start
        self.log.episode('Ended with total reward {} after '.format(total_reward, duration))
        self.ale.reset_game()