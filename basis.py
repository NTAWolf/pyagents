# agent.py

import util
from datetime import datetime
from ale_python_interface import ALEInterface

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

    def __init__(self, game_name, agent, results_dir, use_minimal_action_set=True):
        self.game_name = game_name
        self.agent = agent
        self.use_minimal_action_set = use_minimal_action_set

        now = datetime.now().strftime('%Y%m%d-%H-%M')
        self.results_dir = os.join(results_dir, game_name[:-4] + now) # drop .bin, append current time down to the minute
        self.initiate_results_dir()

        self.log = util.Logger(('settings','action', 'episode','run'), 
                                'episode', os.join(self.results_dir, 'GameManager.log'))

        log.settings("game_name {}".format(game_name))
        log.settings("agent.name {}".format(agent.name))
        log.settings("results_dir {}".format(results_dir))
        log.settings("use_minimal_action_set {}".format(use_minimal_action_set))

    def initiate_results_dir(self):
        os.makedirs(self.results_dir) # Should raise an error if directory exists

    def run(self, n_episodes):
        """Run the wanted number of episodes.
        """
        self.ale = ALEInterface()
        self.ale.loadROM(self.game_name)
        if self.use_minimal_action_set:
            self.actions = self.ale.getMinimalActionSet()
        else:
            self.actions = self.ale.getLegalActionSet()

        self.state_callbacks = (self.get_screen, self.get_screen_grayscale, self.get_screen_RGB, self.get_RAM)


        self.log.run("Starting run for {} episodes".format(n_episodes))
        start = datetime.now()
        for episode in xrange(n_episodes):
            self.log.episode("Starting episode {}".format(episode))
            self._run_episode()
        duration = datetime.now() - start
        self.log.run("Finished run after {}".format(duration))

    def _run_episode(self):
        start = datetime.now()
        total_reward = 0
        n_action = 0
        while not self.ale.game_over():
            state = self.ale.
            action = self.agent.select_action(self.state_callbacks, self.actions)
            reward = self.ale.act(action)
            self.log.action("Action number {}: took action {}, reward {}".format(n_action, action, reward))
            self.agent.receive_reward()
            total_reward += reward
            n_action += 1
        duration = datetime.now() - start
        self.log.episode('Ended with total reward {} after '.format(total_reward, duration))
        self.ale.reset_game()

    # Methods for state perception
    def get_screen(self): 
        """Returns a matrix containing the current game screen."""
        return self.ale.getScreen()

    def get_screen_grayscale(self, grayscale_output_buffer):
        """This method should receive an array of length width × height 
        (generally 160 × 210 = 33, 600) and then it will fill this array 
        with the grayscale colours"""
        return self.ale.getScreenGrayscale(grayscale_output_buffer)

    def get_screen_RGB(self, output_rgb_buffer): 
        """This method should receive an array of length 3× width × height 
        (generally 3 × 160 × 210 = 100, 800) and then it will fill this array 
        with the RGB colours. The first positions contain the red colours, 
        followed by the green colours and then the blue colours"""
        return self.ale.getScreenRGB(output_rgb_buffer)

    def get_RAM(self): 
        """Returns a vector containing current RAM content (byte-level)."""
        return self.ale.getRAM()