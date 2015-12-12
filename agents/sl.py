import numpy as np

from . import Agent
from util.collections import CircularList
from util.managers import RepeatManager, LinearInterpolationManager
from util.nn import MLP
from util.pongcess import RelativeBall, StateIndex


class SLAgent(Agent):
    """Agent using keras NN
    """

    def __init__(self, n_frames_per_action=4):
        super(SLAgent, self).__init__(name='SL', version='1')
        self.experience = CircularList(1000)
        self.epsilon = LinearInterpolationManager([(0, 1.0), (1e4, 0.1)])
        self.action_repeat_manager = RepeatManager(n_frames_per_action - 1)

    def select_action(self):
        # Repeat last chosen action?
        action = self.action_repeat_manager.next()
        if action != None:
            return action

        state = self.preprocessor.process()
        try:
            s = np.array(state).reshape(len(state), 1)
        except:
            s = np.array(state).reshape(1, 1)


        if self._sars[2]:
            self._sars[3] = s
            self.flush_experience()

        # Consider postponing the first training until we have 32 samples
        if len(self.experience) > 0:
            self.nn.train(self.experience)

        if np.random.random() < self.epsilon.next():
            action = self.get_random_action()
        else:
            action_index = self.nn.predict(s)
            action = self.available_actions[action_index]

        self.action_repeat_manager.set(action)

        self._sars[0] = s
        self._sars[1] = self.available_actions.index(action)

        return action

    def set_available_actions(self, actions):
        super(SLAgent, self).set_available_actions(actions)
        # possible state values 
        state_n = len(self.preprocessor.enumerate_states())

        self.nn = MLP(config='simple', input_ranges=[[0, state_n]],
                      n_outputs=len(actions), batch_size=4)

    def set_raw_state_callbacks(self, state_functions):
        self.preprocessor = StateIndex(RelativeBall(state_functions, trinary=True))

    def receive_reward(self, reward):
        self._sars[2] = reward

    def on_episode_start(self):
        self._reset_sars()

    def on_episode_end(self):
        self._sars[3] = self._sars[0]
        self._sars[4] = 0
        self.flush_experience()

    def flush_experience(self):
        self.experience.append(tuple(self._sars))
        self._reset_sars()

    def _reset_sars(self):
        # state, action, reward, newstate, newstate_not_terminal
        self._sars = [None, None, None, None, 1]

    def get_settings(self):
        settings =  {
            "name": self.name,
            "version": self.version,
            "experience_replay": self.experience.capacity(),
            "preprocessor": self.preprocessor.get_settings(),
            "epsilon": self.epsilon.get_settings(),
            "nn": self.nn.get_settings(),
        }

        settings.update(super(SLAgent, self).get_settings())

        return settings
