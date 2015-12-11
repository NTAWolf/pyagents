"""This module contains preprocessing tools for the game Pong.
"""

import numpy as np

import matplotlib.pyplot as plt
import os

def save_image(data, file_path):
    path = os.path.join('fig', file_path)
    aximg = plt.imshow(data)
    plt.savefig(path)
    print "Saved figure in {}".format(path)


BALL_COLOR = 14

BACKGROUND_COLOR = 34

AGENT_COLOR = 200
AGENT_X = 140 # to 143 (incl)

OPPONENT_COLOR = 56
OPPONENT_X = 19 # to 16 (incl)

PLAY_AREA_TOP = 34
PLAY_AREA_BOTTOM = 194

X_RANGE = np.arange(160)
Y_RANGE = np.arange(0, PLAY_AREA_BOTTOM - PLAY_AREA_TOP)


class Feature(object):

    def __init__(self, name):
        self.name = name

    def process(self):
        raise NotImplementedError()

    def enumerate_states(self):
        raise NotImplementedError()

    def get_settings(self):
        """
        Returns a dict
        """
        return {"name":self.name}

def extract_game_area(frame):
    if frame.ndim == 1:
        frame = frame.reshape((210,160))
        return frame[PLAY_AREA_TOP:PLAY_AREA_BOTTOM]
    elif frame.ndim == 3:
        print "RGB frame extract_game_area not tested!"
        return frame[:,PLAY_AREA_TOP:PLAY_AREA_BOTTOM,:]

class StateIndex(Feature):
    """Usage:
    s = StateIndex(RelativeBall())
    s.process(state)
    """
    def __init__(self, feature):
        super(StateIndex, self).__init__('StateIndex')
        self.f = feature
        states = feature.enumerate_states()
        self.state2index = dict([(s,i) for i,s in enumerate(states)])

    def process(self):
        state = self.f.process()
        return self.state2index[state]

    def enumerate_states(self):
        return list(self.state2index.iterkeys())

    def get_settings(self):
        s = super(StateIndex, self).get_settings()
        s.update({'f':self.f.get_settings()})
        return s





class RelativeIntercept(Feature):
    """Returns 1 if ball is expected above,
    -1 if ball is expected below, and 0 if it
    is unknown.
    """

    def __init__(self, raw_state_callbacks):
        super(RelativeIntercept, self).__init__("RelativeIntercept")
        self.pos = Positions(raw_state_callbacks)
        self.last_valid_v = None
        self.last_valid_p = None

        # For debugging
        self.intercept = (5,5)
        self.p1 = -1
        self.ag = -1

        self.LEFT = 0
        self.TOP = 1
        self.RIGHT = 2
        self.BOTTOM = 3


    def process(self):
        self.pos.update()
        p = self.pos.ball

        if p is None:
            return 0

        if self.last_valid_p is None:
            self.last_valid_p = p
            return 0

        v = p - self.last_valid_p
        self.last_valid_p = p

        # Is v corrupt?
        if np.linalg.norm(v[0]) < 1e-1:
            if self.last_valid_v is None:
                return 0
            v = self.last_valid_v
        else:
            self.last_valid_v = v

        # Here, we are guaranteed that p and v will work
        edge = -1
        while edge != self.RIGHT:
            p, v, edge = self.next_intercept(p, v)

        # For debugging
        self.intercept = p + (0, PLAY_AREA_TOP)
        self.p1 = p[1]
        self.ag = self.pos.agent

        # Return relative intercept, clipped to 1,0,-1
        if p[1] < self.pos.agent:
            return 1
        return -1
        # return int(np.clip(diff, -1, 1))

    def next_intercept(self, p, v):
        if v[0] < 0:
            return self.intercept_vertical(p, v, OPPONENT_X, self.LEFT)
        return self.intercept_vertical(p, v, AGENT_X, self.RIGHT)

    def intercept_vertical(self, p, v, x_val, edge):
        # p + a*v = x_val
        # a = (x_val - p) / v
        a = (x_val - p[0]) / v[0]
        i = p + a*v
        if i[1] < 0:
            return self.intercept_horizontal(p, v, 0, self.TOP)
        elif i[1] > 160:
            return self.intercept_horizontal(p, v, 160, self.BOTTOM)
        return i, (-1,1)*v, edge

    def intercept_horizontal(self, p, v, y_val, edge):
        a = (y_val - p[1]) / v[1]
        p = p + a*v
        return p, (1,-1)*v, edge



class RelativeBall(Feature):
    def __init__(self, raw_state_callbacks, trinary=False):
        """if trinary is True, returns -1 if the ball is below the agent,
        0 if the ball is on level with the agent, and 1 if the ball is above
        the agent. I.e. reduces to three states.
        """
        super(RelativeBall, self).__init__('RelativeBall')
        self.pos = Positions(raw_state_callbacks)
        self.trinary = trinary

    def process(self):
        self.pos.update()

        if self.pos.ball[1] is None or self.pos.agent is None:
            return 0
        rel_pos = self.pos.ball[1] - self.pos.agent
        if self.trinary:
            if rel_pos < 0:
                return -1
            if rel_pos > 0:
                return 1
        return rel_pos

    def enumerate_states(self):
        if self.trinary:
            return (-1,0,1)
        return list(set([b-p for p in Y_RANGE for b in Y_RANGE]))

    def get_settings(self):
        s = super(RelativeBall, self).get_settings()
        s.update({'trinary':self.trinary})
        return s


class Positions(Feature):
    """always stores the last valid positions
    If nothing has been seen yet, stores None
    """

    def __init__(self, raw_state_callbacks):
        super(Positions, self).__init__('Positions')
        self.ball = None
        self.agent = None # Agent y position
        self.opponent = None # Opponent y position
        self.f = raw_state_callbacks.raw

    def update(self):
        """Updates the position estimates
        frame is a numpy array with the raw colours from a frame
        """
        frame = extract_game_area(self.f())
        # frame = self.f().reshape((210,160))
        # frame = frame[PLAY_AREA_TOP:PLAY_AREA_BOTTOM]

        self._update_agent(frame)
        self._update_opponent(frame)
        self._update_ball(frame)

    def _update_agent(self, frame):
        res = self.get_mean_pos_1d(frame[:,AGENT_X], AGENT_COLOR)
        if res is not None:
            self.agent = res

    def _update_opponent(self, frame):
        res = self.get_mean_pos_1d(frame[:,OPPONENT_X], OPPONENT_COLOR)
        if res is not None:
            self.opponent = res

    def _update_ball(self, frame):
        hori = np.sum(frame == BALL_COLOR, axis=0)
        vert = np.sum(frame == BALL_COLOR, axis=1)

        x = self.get_mean_pos_1d(hori)
        y = self.get_mean_pos_1d(vert)

        if x is None or y is None:
            return

        self.ball = np.array((x,y))

    def get_mean_pos_1d(self, array, value=None):
        idx = np.arange(len(array), dtype=np.float32)
        if value is not None:
            indices = idx[array == value]
        else:
            # Assume array is a sum on an axis
            indices = idx[array > 0]
        if len(indices) == 0:
            return None
        return np.average(indices)

    def enumerate_states(self):
        """Returns a list of all the possible states
        """
        from itertools import product

        # Agent's possible positions, opponent's possible positions,
        # ball's possible x positions, ball's possible y positions
        return list(product(Y_RANGE, Y_RANGE, X_RANGE, Y_RANGE))

    def get_settings(self):
        return super(Positions, self).get_settings()
