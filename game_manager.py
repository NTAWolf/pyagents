"""This module defines basis classes Agent and GameManager.
Agent is just an interface to be implemented by reinforcement learning agents,
while GameManager is an actual applicable game manager that can be used across
all games and agents supported by ALE.
"""

from collections import namedtuple
from datetime import datetime
import os
from time import time, sleep
import shutil

from ale_python_interface import ALEInterface

import util.logging

ROM_RELATIVE_LOCATION = '../roms/'


class GameManager(object):
    """This class takes care of the interactions between an agent and
    a game across episodes, as well as overall logging of performance.
    """

    def __init__(self, game_name, agent, results_dir,
                 n_epochs=1, n_episodes=None, n_frames=None,
                 remove_old_results_dir=False, use_minimal_action_set=True,
                 min_time_between_frames=0):
        """game_name is one of the supported games (there are many), as a string: "space_invaders.bin"
        agent is an an instance of a subclass of the Agent interface
        results_dir is a string representing a directory in which results and logs are placed
            If it does not exist, it is created.
        use_minimal_action_set determines whether the agent is offered all possible actions,
            or only those (minimal) that are applicable to the specific game.
        min_time_between_frames is the minimum required time in seconds between
            frames. If 0, the game is unrestricted.
        """
        self.game_name = game_name
        self.agent = agent
        self.use_minimal_action_set = use_minimal_action_set
        self.min_time_between_frames = min_time_between_frames
        self.n_epochs = n_epochs
        self.n_episodes = n_episodes
        self.n_frames = n_frames

        if ((n_episodes is None and n_frames is None) or 
            (n_episodes is not None and n_frames is not None)):
            raise ValueError("Extacly one of n_episodes and n_frames "
                             "must be defined")

        
        self.initialize_results_dir(results_dir, remove_old_results_dir)

        self.log = util.logging.Logger(('settings', 'step','episode', 'epoch', 'overall'),
                                       'settings', os.path.join(self.results_dir, 'GameManager.log'))

        self.stats = util.logging.CSVLogger(
                            os.path.join(self.results_dir, 'stats.log'),
                            header='epoch,episode,total_reward,n_frames,wall_time',
                            print_items=True)

        self._object_cache = dict()

        self.initialize_ale()
        self.initialize_agent()

        self.dump_settings()

    def initialize_results_dir(self, results_dir, remove_existing=False):
        """Creates the whole path of directories if they do no exist.
        If they do exist, raises an error unless remove_existing is True,
        in which case the existing directory is deleted.
        """
        now = datetime.now().strftime('%Y%m%d-%H-%M')
        # drop .bin, append current time down to the minute
        results_dir = os.path.join(results_dir, self.game_name[:-4] + now)

        if remove_existing:
            if os.path.exists(results_dir):
                shutil.rmtree(results_dir)
        # Should raise an error if directory exists
        os.makedirs(results_dir)

        self.results_dir = results_dir

    def initialize_ale(self):
        self.ale = ALEInterface()
        self.ale.loadROM(os.path.join(ROM_RELATIVE_LOCATION, self.game_name))

    def initialize_agent(self):
        RSC = namedtuple('RawStateCallbacks', ['raw', 'grey', 'rgb', 'ram'])
        raw_state_callbacks = RSC(self.get_screen, 
                                 self.get_screen_grayscale, 
                                 self.get_screen_RGB, 
                                 self.get_RAM)
        
        self.agent.set_raw_state_callbacks(raw_state_callbacks)
        self.agent.set_results_dir(self.results_dir)
        
        if self.use_minimal_action_set:
            actions = self.ale.getMinimalActionSet()
        else:
            actions = self.ale.getLegalActionSet()

        self.agent.set_available_actions(actions)

    def rest(self, already_elapsed):
        rest_time = self.min_time_between_frames - already_elapsed
        if rest_time > 0:
            sleep(rest_time)

    def run(self):
        """Runs self.n_epochs epochs, where the agent's learning is
        reset for each new epoch.
        Each epoch lasts self.n_episodes or self.n_frames, whichever is 
            defined.
        """
        self.log.overall('Starting run')
        run_start = time()
        for epoch in xrange(self.n_epochs):
            self.agent.reset()
            self.n_epoch = epoch
            self._run_epoch()
        self.log.overall('End of run ({:.2f} s)'.format(time() - run_start))

    def _run_epoch(self):
        self.n_episode = 0

        start = time()
        while not self._stop_condition_met():
            self._run_episode()
            self.n_episode += 1
        wall_time = (time() - start)
        frames = self.ale.getFrameNumber()

        self.log.epoch("Finished epoch after {:.2f} seconds".format(wall_time))

    def _run_episode(self):
        self.ale.reset_game()
        self.agent.on_episode_start()

        total_reward = 0
        episode_start = time()

        while (not self.ale.game_over()) and (not self._stop_condition_met()):
            timestep_start = time()

            action = self.agent.select_action()
            reward = self.ale.act(action)
            self.agent.receive_reward(reward)

            total_reward += reward

            self.rest(time() - timestep_start)

        wall_time = time() - episode_start
        self.agent.on_episode_end()


        # Stats format: CSV with epoch, episode, total_reward, n_frames, wall_time
        self.stats.write(self.n_epoch, self.n_episode, total_reward, 
                         self.ale.getEpisodeFrameNumber(), '{:.2f}'.format(wall_time))

    def _stop_condition_met(self):
        if self.n_episodes:
            return self.n_episode >= self.n_episodes
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
            "n_epochs": self.n_epochs,
            "n_episodes": self.n_episodes,
            "n_frames": self.n_frames,
            "agent": self.agent.get_settings(),
            "results_dir": self.results_dir,
            "use_minimal_action_set": self.use_minimal_action_set,
        }
