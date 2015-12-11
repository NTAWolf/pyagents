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


class BallIntercept(Feature):
    """Based on the ball's position last time process was called,
    this class predicts where the ball will intercept with the 
    agent's paddle space.
    """
    def __init__(self, raw_state_callbacks):
        super(BallIntercept, self).__init__('BallIntercept')
        self.pos = Positions(raw_state_callbacks)
        self.play_area_size = np.array([AGENT_X - OPPONENT_X,
                                        PLAY_AREA_BOTTOM - PLAY_AREA_TOP],
                                        dtype=np.float32)
        self.last_ball = self.play_area_size / 2
        self.LEFT = 0
        self.TOP = 1
        self.RIGHT = 2
        self.BOTTOM = 3

    def process(self):
        """Returns an integer
        """
        self.pos.update()
        ball = np.array(self.pos.ball, dtype=np.float32)

        vel = ball - self.last_ball
        self.last_ball = ball.copy()
        
        edge = -1
        while edge != self.RIGHT:
            ball, vel, edge = self.next_intercept(ball, vel)

        return int(ball[1])

    def next_intercept(self, pos, vel):
        """Returns edge intercept point and reflected velocity
        around edge normal, as well as intercept edge constant
        """
        reflected_vel = vel * (-1, 1)
        # Moving left?
        if vel[0] < 0:
            # a is the multiplier on vel to achieve left side intercept
            a = (OPPONENT_X - pos[0])/vel[0]
            intercept = pos + a*vel
            # Do we hit the left edge?
            if PLAY_AREA_TOP <= intercept[1] <= PLAY_AREA_BOTTOM:
                return intercept, reflected_vel, self.LEFT
            return self.top_bottom_intercept(pos, vel)

        # Otherwise, moving right
        a = (AGENT_X - pos[0])/vel[0]
        intercept = pos + a*vel
        # Do we hit the right edge?
        if PLAY_AREA_TOP <= intercept[1] <= PLAY_AREA_BOTTOM:
            return intercept, reflected_vel, self.RIGHT
        return self.top_bottom_intercept(pos, vel)

    def top_bottom_intercept(self, pos, vel):
        """Returns the next intercept with a top layer.
        Only used when you most definitely know you'll not
        hit a side edge.

        Returns edge intercept point, reflected velocity
        around edge normal, and edge constant
        """
        reflected_vel = vel * (1, -1)
        # Moving down? (y-axis is inverted)
        if vel[1] > 0:
            # a is the multiplier on vel to achieve bottom edge intercept
            a = (PLAY_AREA_BOTTOM - pos[1])/vel[1]
            intercept = pos + a*vel
            return intercept, reflected_vel, self.BOTTOM
        # a is the multiplier on vel to achieve top edge intercept
        a = (PLAY_AREA_TOP - pos[1])/vel[1]
        intercept = pos + a*vel
        return intercept, reflected_vel, self.TOP

    def get_settings(self):
        return super(BallIntercept, self).get_settings()


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

    def __init__(self, raw_state_callbacks):
        super(Positions, self).__init__('Positions')
        self.ball = (0,0)
        self.agent = 0 # Agent y position
        self.opponent = 0 # Opponent y position
        self.f = raw_state_callbacks.raw

    def update(self):
        """Updates the position estimates
        frame is a numpy array with the raw colours from a frame
        """
        frame = self.f().reshape((210,160))
        frame = frame[PLAY_AREA_TOP:PLAY_AREA_BOTTOM]    

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
        # try:
        x = self.get_mean_pos_1d(hori)
        y = self.get_mean_pos_1d(vert)
        if x is None:
            x = self.ball[0]
        if y is None:
            y = self.ball[1]
        self.ball = (x,y)

    def get_mean_pos_1d(self, array, value=None):
        idx = np.arange(len(array), dtype=np.uint8)
        if value is not None:
            indices = idx[array == value]
        else:
            # Assume array is a sum on an axis
            indices = idx[array > 0]
        if len(indices) == 0:
            return None
        return int(np.average(indices))

    def enumerate_states(self):
        """Returns a list of all the possible states
        """
        from itertools import product

        # Agent's possible positions, opponent's possible positions,
        # ball's possible x positions, ball's possible y positions
        return list(product(Y_RANGE, Y_RANGE, X_RANGE, Y_RANGE))

    def get_settings(self):
        return super(Positions, self).get_settings()
