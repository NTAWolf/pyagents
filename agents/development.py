import threading

from . import Agent


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()


def check_stop_thread():
    if threading.current_thread().stopped():
        raise StopIteration


class DevelopmentAgent(Agent):
    """This agent is for development purposes.
    Strategically placed breakpoints give the developer access to the same 
    knowledge and actions as the agent has."""

    def __init__(self, storage, thread_event):
        """storage is a dict used for communication between this agent and a
        development application. Variables are stored there.
        """
        super(DevelopmentAgent, self).__init__(
            name='DevelopmentAgent', version='1')
        self.storage = storage
        self.thread_event = thread_event

        print "Note that the step-through does not work"

    def select_action(self, state, available_actions):
        """Returns one of the actions given in available_actions.
        """
        self.storage['state'] = state
        self.storage['available_actions'] = available_actions
        # pdb.set_trace()
        print "About to wait for thread_event..."
        # Ipython regains control here
        self.thread_event.wait()
        self.thread_event.clear()

        print "Past thread_event"
        check_stop_thread()
        print "Thread not supposed to stop. Moving on."
        return self.storage.get('action', 0)

    def receive_reward(self, reward):
        self.storage['reward'] = reward

    def on_episode_start(self):
        print "Agent sees that episode starts"

    def on_episode_end(self):
        print "Agent sees that episode ends"

    def get_printable_settings(self):
        return dict()
