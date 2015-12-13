import numpy as np

from . import Agent
from util.collections import CircularList
from util.managers import RepeatManager, LinearInterpolationManager
from util.pongcess import RelativeIntercept, StateIndex

class SarsaAgent(Agent):
    """
    Agent that uses a SARSA(lambda)
    Input RGB image is preprocessed, resulting in states
    - (x, y) ball
    - y player
    - y opponent
    """


    def __init__(self, n_frames_per_action=4, 
                 trace_type='replacing', 
                 learning_rate=0.001,
                 discount=0.99, 
                 lambda_v=0.5,
                 record=False):
        super(SarsaAgent, self).__init__(name='Sarsa', version='1')
        self.n_frames_per_action = n_frames_per_action

        self.epsilon = LinearInterpolationManager([(0, 1.0), (1e4, 0.1)])
        self.action_repeat_manager = RepeatManager(n_frames_per_action - 1)
        
        self.trace_type = trace_type
        self.learning_rate = learning_rate
        self.lambda_v = lambda_v
        self.discount = discount

        self.a_ = 0
        self.s_ = 0
        self.r_ = 0

        self.q_vals = None
        self.e_vals = None

        self.n_goals = 0
        self.n_greedy = 0
        self.n_random = 0

        self.record = record
        if record:
            # 5 action, 3 states 
            # => q_vals.shape == (5, 3)
            #    e_vals.shape == (5, 3)
            #    sarsa.shape == (5, 1)
            self.mem = CircularList(10000) 

    def reset(self):
        pass

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
        r = self.r_
        # print "running SARSA with {}".format([s, a, r, s_, a_])

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

        # save current state, action for next iteration
        self.s_ = s_
        self.a_ = a_

        self.r_ = 0

        # save the state
        if self.record: 
            self.mem.append({'e_vals': np.copy(self.e_vals), 
                             'q_vals': np.copy(self.q_vals), 
                             'sarsa': (s, a, r, s_, a_)})

        return self.available_actions[a_]

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
        self.n_greedy += 1
        return action

    def set_available_actions(self, actions):
        super(SarsaAgent, self).set_available_actions(actions)
        # possible state values 
        print 'type(actions)',type(actions)
        state_n = len(self.preprocessor.enumerate_states())

        print 'state_n',state_n
        print 'actions',actions
        self.q_vals = np.zeros((state_n, len(actions)))
        self.e_vals = np.zeros((state_n, len(actions)))

    def set_raw_state_callbacks(self, state_functions):
        self.preprocessor = RelativeIntercept(state_functions)

    def receive_reward(self, reward):
        # TODO: receive_rewards() called too frequently! self.r_ is changed
        #       before select_action completes!
        self.r_ += reward
        if reward > 0:
            self.n_goals += 1

    def on_episode_start(self):
        self.n_goals = 0
        self.n_greedy = 0
        self.n_random = 0

    def on_episode_end(self):
        print "  q(s): {}".format(self.q_vals)
        print "  e(s): {}".format(self.e_vals)
        print "  goals: {}".format(self.n_goals)
        print "  n_greedy: {}".format(self.n_greedy)
        print "  n_random: {}".format(self.n_random)

        if self.record:
            a_s = [(e['sarsa'][4], e['sarsa'][3]) for e in self.mem]
            a_counts = [0] * self.q_vals.shape[0]
            s_counts = [0] * self.q_vals.shape[1]
            for a, s in a_s:
                a_counts[a] += 1
                s_counts[s] += 1
            print "  actions: {}".format(a_counts)
            print "  states: {}".format(s_counts)

            self.mem.clear()

    def get_learning_dump(self):
        return self.mem

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

        settings.update(super(SarsaAgent, self).get_settings())
        
        return settings
