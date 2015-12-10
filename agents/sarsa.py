import numpy as np

from . import Agent
from util.collections import CircularList
from util.managers import RepeatManager, LinearInterpolationManager
from util.pongcess import RelativeBall, StateIndex

class SarsaAgent(Agent):
    """
    Agent a SARAS(lambda) approach
    Input RGB image is preprocessed, resulting in states
    - (x, y) ball
    - y player
    - y opponent
    """

    def __init__(self, n_frames_per_action=4, trace_type='replacing', 
                 learning_rate=0.001,
                 discount=0.99, 
                 lambda_v=0.5):
        super(DLAgent, self).__init__(name='DL', version='1')
        self.experience = CircularList(1000)
        self.n_frames_per_action = n_frames_per_action

        self.epsilon = LinearInterpolationManager([(0, 1.0), (1e4, 0.1)])
        self.action_repeat_manager = RepeatManager(n_frames_per_action - 1)
        
        self.preprocessor = StateIndex(RelativeBall(self.raw_state))

        self.trace_type = trace_type
        self.learning_rate = learning_rate
        self.lambda_v = lambda_v
        self.discount = discount

        self.a_ = 0
        self.s_ = 0
        self.r_ = 0

        self.q_vals = None
        self.e_vals = None

    def get_most_valuable_action(self):
        return np.argmax(self.q_vals)

    def select_action(self):
        """
        Initialize Q(s; a) arbitrarily, for all s in S; a in A(s)
        Repeat (for each episode):
            E(s; a) = 0, for all s 2 S; a 2 A(s)
            Initialize S, A
            Repeat (for each step of episode):
              S = S'; A = A'
              Take action A, observe R, S'
              Choose A' from S' using policy derived from Q (e.g., e-greedy)
              update_q()
            until S is terminal
        """
        sid = self.preprocessor.process()

        action = self.action_repeat_manager.next()
        if action != None:
            return action

        # assign previous s' to the current s
        s = self.s_
        # assign previous a' to the current a
        a = self.a_
        # get current state
        s_ = sid

        a_ = self.e_greedy(s_)
        self.action_repeat_manager.set(a_)

        print "running SARSA with {}".format([s, a, r, s_, a_])

        """
              d = R + gamma*Q(S', A') - Q(S, A)
              E(S,A) = E(S,A) + 1           (accumulating traces)
           or E(S,A) = (1 - a) * E(S,A) + 1 (dutch traces)
           or E(S;A) = 1                    (replacing traces)
              For all s in S; a in A(s):
                Q(s,a) = Q(s,a) + E(s,a)   
                E(s,a) = gamma * lambda * E(s,a)
        """
        d = r + self.discount * self.q_vals[s_, a_] - self.q_vals[s, a]
        if self.trace_type is 'accumulating':
            self.e_vals[s,a] += 1
        elif self.trace_type is 'replacing':
            self.e_vals[s,a] = 1
        elif self.trace_type is 'dutch':
            self.e_vals[s,a] *= (1 - self.learning_rate)
            self.e_vals[s,a] += 1
            
        # TODO: currently Q(s, a) is updated for all a, not a in A(s)!
        self.q_vals += self.learning_rate * d * self.e_vals
        self.e_vals *= (self.discount * self.lambda_v)
        
        print "  q(s) {}".format(self.q_vals[s,:])
        print "  e(s) {}".format(self.e_vals[s,:])

        # save current state, action for next iteration
        self.s_ = s_
        self.a_ = a_

        return a_

    def e_greedy(self, sid):
        # decide on next action a'
        # E-greedy strategy
        if np.random.random() < self.epsilon.next(): 
            action = self.get_random_action()
            print "cheeky pint? action {}".format(action)
        else:
            # get the best action given the current state
            action = np.argmax(self.q_vals[sid, :])
            print "greedy bastard {}".format(action)
        return action

    def set_available_actions(self, actions):
        # possible state values 
        state_n = len(self.preprocessor.enumerate_states())

        self.q_vals = np.random.rand((state_n, len(actions)))
        self.e_vals = np.zeros((state_n, len(actions)))

    def receive_reward(self, reward):
        self.r = reward

    def on_episode_end(self):
        self.flush_experience()

    def flush_experience(self):
        self.experience.append(tuple(self._sars))

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
