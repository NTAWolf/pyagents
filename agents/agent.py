
class Agent(object):
    """This class defines the agent interface to be used in this project.
    """

    def __init__(self, name, version):
        self.name = name
        self.version = version

    def select_action(self, state, available_actions):
        """Returns one of the actions given in available_actions.
        """
        raise NotImplementedError("Method not implemented")

    def receive_reward(self, reward):
        raise NotImplementedError("Method not implemented")

    def on_episode_start(self):
        """Called on episode start by the GameManager
        """
        raise NotImplementedError("Method not implemented")

    def on_episode_end(self):
        """Called on episode end by the GameManager
        """
        raise NotImplementedError("Method not implemented")
