import numpy as np

from . import Agent
from util.collections import CircularList
from util.managers import RepeatManager, LinearInterpolationManager
from util.pongcess import RelativeIntercept, StateIndex
from util.logging import CSVLogger


class Sarsa2Agent(Agent):
    """
    Agent that uses a SARSA(lambda)
    Input RAW image is preprocessed, resulting in states
    + Predicted ball position above player
    + Predicted ball position within the player pad
    + Predicted ball position beneath player
    """


    def __init__(self, n_frames_per_action=4, 
                 trace_type='replacing', 
                 learning_rate=0.001,
                 discount=0.99, 
                 lambda_v=0.5):
        super(Sarsa2Agent, self).__init__(name='Sarsa2', version='2')
        self.n_frames_per_action = n_frames_per_action

        self.epsilon = LinearInterpolationManager([(0, 1.0), (1e4, 0.005)])
        self.action_repeat_manager = RepeatManager(n_frames_per_action - 1)
        
        self.trace_type = trace_type
        self.learning_rate = learning_rate
        self.lambda_v = lambda_v
        self.discount = discount

        self.q_vals = None
        self.e_vals = None

        self.initialize_asr_and_counters()

    def initialize_asr_and_counters(self):
        self.a_ = 0
        self.s_ = 0
        self.r_ = 0

        self.n_goals = 0
        self.n_greedy = 0
        self.n_random = 0
        self.n_rr = 0
        self.n_sa = 0

        self.n_episode = 0

    def reset(self):
        self.q_vals[:] = 0.0
        self.e_vals[:] = 0.0
        self.epsilon.reset()
        self.initialize_asr_and_counters()

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
        self.n_sa += 1

        sid = self.preprocessor.process()

        # assign previous s' to the current s
        s = self.s_
        # assign previous a' to the current a
        a = self.a_
        # get current state
        s_ = self.state_mapping[str(sid)]

        r = self.r_

        # select action:
        # - repeat previous action based on the n_frames_per_action param
        # - OR choose an action according to the e-greedy policy 
        a_ = self.action_repeat_manager.next()
        if a_ is None:
            a_ = self.e_greedy(s_)
            self.action_repeat_manager.set(a_)

        # Calculate update delta
        d = r + self.discount * self.q_vals[s_, a_] - self.q_vals[s, a]

        # Handle traces
        self.update_trace(s,a)

        # TODO: currently Q(s, a) is updated for all a, not a in A(s)!
        self.q_vals += self.learning_rate * d * self.e_vals
        self.e_vals *= (self.discount * self.lambda_v)

        # save current state, action for next iteration
        self.s_ = s_
        self.a_ = a_

        # save the state
        self.rlogger.write(self.n_episode, 
                           *[q for q in list(self.q_vals.flatten())
                             + list(self.e_vals.flatten())])

        return self.available_actions[a_]

    def set_results_dir(self, results_dir):
        super(Sarsa2Agent, self).set_results_dir(results_dir)

    def update_trace(self, s, a):
        if self.trace_type is 'accumulating':
            self.e_vals[s,a] += 1
        elif self.trace_type is 'replacing':
            self.e_vals[s,a] = 1
        elif self.trace_type is 'dutch':
            self.e_vals[s,a] *= (1 - self.learning_rate)
            self.e_vals[s,a] += 1

    def e_greedy(self, sid):
        """Returns action index
        """
        # decide on next action a'
        # E-greedy strategy
        if np.random.random() < self.epsilon.next(): 
            action = self.get_random_action()
            action = np.argmax(self.available_actions == action)
            self.n_random += 1
            # get the best action given the current state
        else:
            action = np.argmax(self.q_vals[sid, :])
            #print "greedy action {} from {}".format(action, self.q_vals[sid,:])
            self.n_greedy += 1
        return action

    def set_available_actions(self, actions):
        super(Sarsa2Agent, self).set_available_actions(actions)

        states = self.preprocessor.enumerate_states()
        state_n = len(states)

        # generate state to q_val index mapping
        self.state_mapping = dict([('{}'.format(v), i) 
                                    for i, v in enumerate(states)])
        print "Agent state_mapping:", self.state_mapping

        print 'state_n',state_n
        print 'actions',actions
        self.q_vals = np.zeros((state_n, len(actions)))
        self.e_vals = np.zeros((state_n, len(actions)))

        headers = 'episode'
        for q in range(len(self.q_vals.flatten())):
            headers += ',q{}'.format(q)
        for e in range(len(self.e_vals.flatten())):
            headers += ',e{}'.format(e)
        self.rlogger = CSVLogger(self.results_dir + '/q_e.csv', 
                                 headers, print_items=False)


    def set_raw_state_callbacks(self, state_functions):
        self.preprocessor = RelativeIntercept(state_functions)

    def receive_reward(self, reward):
        #print "receive_reward {}".format(self.n_rr)
        self.n_rr += 1
        self.r_ = reward
        if reward > 0:
            self.n_goals += 1

    def on_episode_start(self):
        self.n_goals = 0
        self.n_greedy = 0
        self.n_random = 0

    def on_episode_end(self):
        self.n_episode += 1
        #print "  q(s): {}".format(self.q_vals)
        #print "  e(s): {}".format(self.e_vals)
        #print "  goals: {}".format(self.n_goals)
        #print "  n_greedy: {}".format(self.n_greedy)
        #print "  n_random: {}".format(self.n_random)


    def get_settings(self):
        settings =  {
            "name": self.name,
            "version": self.version,
            "preprocessor": self.preprocessor.get_settings(),
            "n_frames_per_action": self.n_frames_per_action,
            "learning_rate": self.learning_rate,
            "discount_rate": self.discount, 
            "lambda": self.lambda_v,
        }

        settings.update(super(Sarsa2Agent, self).get_settings())
        
        return settings
