import numpy as np

from . import Agent
from util.collections import CircularList
from util.managers import RepeatManager, LinearInterpolationManager
from util.pongcess import RelativeIntercept, StateIndex
from util.logging import CSVLogger


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

        self.epsilon = LinearInterpolationManager([(0, 1.0), (1e4, 0.005)])
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
            self.mem = CircularList(100000) 

        self.n_rr = 0
        self.n_sa = 0

        self.n_episode = 0


    def reset(self):
        pass

    def select_action(self):
        #print "select_action {}".format(self.n_sa)
        self.n_sa += 1

        #if self.n_sa > 20:
        #import sys
        #sys.exit(0)
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

        # assign previous s' to the current s
        s = self.s_
        # assign previous a' to the current a
        a = self.a_
        # get current state
        s_ = sid

        r = self.r_

        # select action:
        # - repeat previous action based on the n_frames_per_action param
        # - OR choose an action according to the e-greedy policy 
        a_ = self.action_repeat_manager.next()
        if a_ is None:
            a_ = self.e_greedy(s_)
            self.action_repeat_manager.set(a_)

        #print "running SARSA with {}".format([s, a, r, s_, a_])

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

        #if r != 0:
        #    print "lr: {} d: {}".format(self.learning_rate, d)
        #    print "d q_vals\n{}".format(self.q_vals - p_q_vals)


        # save current state, action for next iteration
        self.s_ = s_
        self.a_ = a_

        # save the state
        self.rlogger.write(self.n_episode, *[q for q in self.q_vals.flatten()])

        if self.record: 
            self.mem.append({'q_vals': np.copy(self.q_vals), 
                             'sarsa': (s, a, r, s_, a_)})

        return self.available_actions[a_]

    def set_results_dir(self, results_dir):
        super(SarsaAgent, self).set_results_dir(results_dir)

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
        super(SarsaAgent, self).set_available_actions(actions)
        # possible state values 
        state_n = len(self.preprocessor.enumerate_states())

        print 'state_n',state_n
        print 'actions',actions
        self.q_vals = np.zeros((state_n, len(actions)))
        self.e_vals = np.zeros((state_n, len(actions)))

        headers = 'episode'
        for q in range(len(self.q_vals.flatten())):
            headers = headers + ',q{}'.format(q)
        self.rlogger = CSVLogger(self.results_dir + '/agent.csv', 'episode,q1,q2,q3,q4,q5,q6,q7,q8,q9', print_items=False)


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
