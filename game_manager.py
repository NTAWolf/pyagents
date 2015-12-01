"""This module defines basis classes Agent and GameManager.
Agent is just an interface to be implemented by reinforcement learning agents,
while GameManager is an actual applicable game manager that can be used across
all games and agents supported by ALE.
"""

from collections import namedtuple
from datetime import datetime
import os
from threading import Thread
import shutil
import util.logging

from ale_python_interface import ALEInterface

from visualise import Visualiser

ROM_RELATIVE_LOCATION = '../roms/'


class GameManager(object):
    """This class takes care of the interactions between an agent and
    a game across episodes, as well as overall logging of performance.
    """

    def __init__(self, game_name, agent, results_dir,
                 remove_old_results_dir=False, use_minimal_action_set=True, 
                 visualise=None):
        """game_name is one of the supported games (there are many), as a string: "space_invaders.bin"
        agent is an an instance of a subclass of the Agent interface
        results_dir is a string representing a directory in which results and logs are placed
            If it does not exist, it is created.
        use_minimal_action_set determines whether the agent is offered all possible actions,
            or only those (minimal) that are applicable to the specific game.
        visualise is None for no visualization (default), or one of 'raw', 'ram', 'grey', 'rgb',
            or a method that takes as an argument the GameManager's list of methods (its 
            state_functions) and returns a new method that returns an np.array for Visualiser.
            The RAM vector is reshaped to a 8x16 array.
        """
        self.game_name = game_name
        self.agent = agent
        self.use_minimal_action_set = use_minimal_action_set
        self.visualise = visualise

        now = datetime.now().strftime('%Y%m%d-%H-%M')
        # drop .bin, append current time down to the minute
        self.results_dir = os.path.join(results_dir, game_name[:-4] + now)
        self.initialize_results_dir(remove_old_results_dir)

        self.log = util.logging.Logger(('settings', 'action', 'episode', 'run'),
                                       'episode', os.path.join(self.results_dir, 'GameManager.log'))

        self._object_cache = dict()

    def initialize_results_dir(self, remove_existing=False):
        """Creates the whole path of directories if they do no exist.
        If they do exist, raises an error unless remove_existing is True,
        in which case the existing directory is deleted.
        """
        if remove_existing:
            if os.path.exists(self.results_dir):
                shutil.rmtree(self.results_dir)
        # Should raise an error if directory exists
        os.makedirs(self.results_dir)

    def initialize_visualiser(self):
        """If the internal flag for visualization is set, 
        prepare the visualizer.
        """
        if self.visualise:
            framerate = 60
            if self.visualise == 'ram':
                def callback():
                    ram = self.get_RAM()
                    return ram.reshape((8, -1))
            elif self.visualise == 'raw':
                callback = self.get_screen
            elif self.visualise == 'grey':
                callback = self.get_screen_grayscale
            elif self.visualise == 'rgb':
                callback = self.get_screen_RGB
            else:
                callback = self.visualise(self.state_functions)

            self.visualiser = Visualiser(callback, framerate,
                                         title="{}: {}".format(self.game_name, self.visualise))
        else:
            self.visualiser = None

    def initialize_run(self, n_episodes, n_frames):
        if n_episodes == n_frames:
            self.log.run("Aborted due to bad input to run()")
            raise ValueError(
                "One and only one of n_episodes and n_frames can be defined at a time")

        self.n_episodes = n_episodes
        self.n_frames = n_frames

        self.log.settings("n_episodes {}".format(str(n_episodes)))
        self.log.settings("n_frames {}".format(str(n_frames)))

        self.ale = ALEInterface()
        self.ale.loadROM(os.path.join(ROM_RELATIVE_LOCATION, self.game_name))
        if self.use_minimal_action_set:
            actions = self.ale.getMinimalActionSet()
        else:
            actions = self.ale.getLegalActionSet()

        self.agent.set_available_actions(actions)

        SF = namedtuple('StateFunctions', ['raw', 'grey', 'rgb', 'ram'])
        self.state_functions = SF(
            self.get_screen, self.get_screen_grayscale, self.get_screen_RGB, self.get_RAM)
        self.episodes_passed = 0

        self.initialize_visualiser()
        self.dump_settings()
        

    def run(self, n_episodes=None, n_frames=None):
        """Run the wanted number of episodes or the wanted number of frames. 
        No more than one of them can be assigned to a value at a time.
        """
        self.initialize_run(n_episodes, n_frames)
        if self.visualiser:
            t = Thread(target=self._run)
            t.start()
            self.visualiser.run()
            # self.visualiser.on_draw(None)
        else:
            self._run()

    def _run(self):
        """Run the wanted number of episodes or the wanted number of frames. 
        No more than one of them can be assigned to a value at a time.
        """
        self.log.run("Starting run")
        start = datetime.now()
        while not self._stop_condition_met():
            self.log.episode(
                "Starting episode {}".format(self.episodes_passed))
            self._run_episode()
            self.episodes_passed += 1
        duration = (datetime.now() - start).total_seconds()
        self.log.run("Finished run after {} seconds".format(duration))

    def _run_episode(self):
        start = datetime.now()
        total_reward = 0
        nframes = 0

        self.agent.on_episode_start()
        while (not self.ale.game_over()) and (not self._stop_condition_met()):
            action = self.agent.select_action(self.state_functions)
            reward = self.ale.act(action)
            self.agent.receive_reward(reward)
            total_reward += reward
            nframes += 1
        self.agent.on_episode_end()

        duration = (datetime.now() - start).total_seconds()
        self.log.episode('Ended with total reward {} after {} seconds and {} frames'.format(
            total_reward, duration, nframes))
        self.ale.reset_game()

    def _stop_condition_met(self):
        if self.n_episodes:
            return self.episodes_passed >= self.n_episodes
        return self.ale.getFrameNumber() >= self.n_frames

    # Methods for state perception
    def get_screen(self):
        """Returns a matrix containing the current game screen in raw pixel data,
        i.e. before conversion to RGB. Handles reuse of np.array object, so it 
        will overwrite what is in the old object"""
        return self._cached('raw', self.ale.getScreen)

    def get_screen_grayscale(self):
        """Returns an np.array with the screen grayscale colours. 
        Handles reuse of np.array object, so it will overwrite what 
        is in the old object.
        """
        return self._cached('gray', self.ale.getScreenGrayscale)

    def get_screen_RGB(self):
        """Returns a numpy array with the screen's RGB colours. 
        The first positions contain the red colours, followed by
        the green colours and then the blue colours"""
        return self._cached('rgb', self.ale.getScreenRGB)

    def get_RAM(self):
        """Returns a vector containing current RAM content (byte-level).
        Handles reuse of np.array object, so it will overwrite what 
        is in the old object"""
        return self._cached('ram', self.ale.getRAM)

    def _cached(self, key, func):
        if key in self._object_cache:
            func(self._object_cache[key])
        else:
            self._object_cache[key] = func()

        return self._object_cache[key]

    def dump_settings(self):
        import json

        settings = self.get_settings()
        path = os.path.join(self.results_dir, 'settings')
        with open(path, 'w') as f:
            json.dump(settings, f, indent=4)

    def get_settings(self):
        """Returns a dict representing the settings needed to 
        reproduce this object and its subobjects
        """
        return {
            "game_name": self.game_name,
            "agent": self.agent.get_settings(),
            "results_dir": self.results_dir,
            "use_minimal_action_set": self.use_minimal_action_set,
            "visualise": str(self.visualise),
        }
