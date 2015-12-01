import numpy as np

from . import Agent
from util.collections import CircularList
from util.managers import RepeatManager, LinearInterpolationManager
from util.nn import Preprocessor, DummyCNN, CNN


class DLAgent(Agent):
    """Agent using keras CNN
    Preprocesses input rgb frame:
        Take max for each channel over this and last frames
        Convert to luminance
        Scale down to 64x64 pixels
        Concatenate with the last three preprocessed inputs

    Learning is done in a CNN:
        Input layer:
            64x64x4
        1st hidden: 
            32 filters of 8x8, stride 4
            Rectifier
        2nd hidden:
            64 filters of 4x4, stride 2
            Rectifier
        3rd hidden:
            64 filters of 3x3, stride 1
            Rectifier
        4th hidden:
            512 fully-connected rectifier units
        Output layer:
            4-18 fully connected linear units

    Selects an action only every four frames; otherwise repeats last selected action

    Uses variable e-greedy: From frame 0 to frame 1e4, it interpolates from e=1 to e=0.1
    """

    def __init__(self, n_frames_per_action=4):
        super(DLAgent, self).__init__(name='DL', version='1')
        self.experience = CircularList(1000)
        self.n_frames_per_action = n_frames_per_action
        self.preprocessor = Preprocessor(scale_shape=(64, 64),
                                         n_frame_concat=3, n_frame_max=1)
        self.epsilon = LinearInterpolationManager([(0, 1.0), (1e4, 0.1)])
        self.action_repeat_manager = RepeatManager(n_frames_per_action - 1)

        self._sar = None  # state, action, reward

    def select_action(self, state):
        # Repeat last chosen action?
        action = self.action_repeat_manager.next()
        if action != None:
            return action

        s = self.preprocessor.process(state)

        if self._sars[2]:
            self._sars[3] = s
            self.flush_experience()

        # Consider postponing the first training until we have 32 samples
        if len(self.experience) > 0:
            self.cnn.train(self.experience)

        if np.random.random() < self.epsilon.next():
            action = self.get_random_action()
        else:
            action_index = self.cnn.predict(s)
            action = self.available_actions[action_index]

        self.action_repeat_manager.set(action)

        self._sars[0] = s
        self._sars[1] = action

        return action

    def set_available_actions(self, actions):
        super(DLAgent, self).set_available_actions(actions)
        self.cnn = CNN(config='deepmind', n_outputs = len(actions))
        # self.cnn = DummyCNN(len(actions))

    def receive_reward(self, reward):
        self._sars[2] = reward

    def on_episode_start(self):
        self._reset_sars()

    def on_episode_end(self):
        self.flush_experience()

    def flush_experience(self):
        self.experience.append(tuple(self._sars))
        self._reset_sars()

    def _reset_sars(self):
        # state, action, reward, newstate
        self._sars = [None, None, None, None]

    def get_settings(self):
        settings =  {
            "name": self.name,
            "version": self.version,
            "n_frames_per_action": self.n_frames_per_action,
            "experience_replay": self.experience.capacity(),
            "preprocessor": self.preprocessor.get_settings(),
            "epsilon": self.epsilon.get_settings(),
            "cnn": self.cnn.get_settings(),
        }

        settings.update(super(DLAgent, self).get_settings())

        return settings
