from random import randrange
import time

import numpy as np
import neurolab as nl

from .collections import CircularList

class NN(object):
    """
    Base class for neural network providing a uniform interface to build, 
    train and evaluate various networks.
    """
    def __init__(self, name, version):
        self.name = name
        self.version = version

    def train(self, td):
        raise NotImplementedError()

    def predict(self, state):
        raise NotImplementedError()

    def get_settings(self):
        return {
            "name": self.name,
            "version": self.version,
        }



# Rough interface estimate. Up for change.
class DummyNN(NN):
    def __init__(self, n_outputs):
        self.n_outputs = n_outputs

    def train(self, memory):
        pass

    def predict(self, state):
        return randrange(self.n_outputs)
        
class MLP(NN):
    """
    Convolutional Neural Networks! Yay
    """
    def __init__(
            self, input_ranges=[[0, 1]], n_outputs=3,
            config='simple', gamma=0.99,
            batch_size=1):
        super(MLP, self).__init__(name='MLP', version='1')

        self.gamma = gamma
        self.iteration = 0
        self.batch_size = batch_size

        self.net = nl.net.newff(input_ranges,[1, n_outputs])


    def train(self, memory):
        """
        memory is a list of state-action-reward-state' tuples
        """

        self.iteration += 1
        train_err = 0
        start_time = time.time()

        # sample random minibatch from memory
        # Should not raise ValueError anymore, as train is only called
        # when memory has content.
        samples = memory.uniform_random_sample(self.batch_size)

        # TODO: this probably isn't very efficient...?
        s, a, r, s_n, t = zip(*samples)

        # print "sample={}".format(s)
        # print "actions={}".format(a)
        # print "rewards={}".format(r)
        # print "next_sample={}".format(s_n)

        target = self.make_target(s, a, r, s_n, t)

        error = self.net.train(s[0], target, epochs=1)

        print "training loss: {}".format(error)

        return error

    def make_target(self, s, a, r, s_, a_):
        return np.array([0, 0, 0]).reshape(1, 3)

    def predict(self, state):
        out = self.net.sim(state)
        return np.argmax(out)

    def get_settings(self):
        settings = {
            "gamma": self.gamma,
#            "network": self.net,
        }

        settings.update(super(MLP, self).get_settings())

        return settings

