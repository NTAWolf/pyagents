from random import randrange
import time

import lasagne as nom
import numpy as np
from scipy.misc import imresize
import theano.tensor as T
import theano

from .collections import CircularList


class Preprocessor(object):
    """This class encapsulates preprocessing settings
    and actions for the following operations:
    - Take max for each color channel over this and 
        previous n_frame_max frames
    - Convert to luminance (using luma constants)
    - Scale down to SCALE_SHAPE pixels
    - Concatenate with the last n_frame_concat preprocessed inputs
    """

    def __init__(self, scale_shape, n_frame_concat, n_frame_max):
        """scale_shape is the wanted shape of the output's first two 
            dimensions
        n_frame_concat is the number of preprocessed frames that are
            to be concatenated with the current frame.
        n_frame_max is the number of previous frames over which we
            take the max value for each channel.
        """
        self.scale_shape = scale_shape
        # Holds the last n_frame_max unprocessed frames
        self._unprocessed = CircularList(n_frame_max)
        # Holds the last n_frame_concat processed frames
        self._processed = CircularList(n_frame_concat)

    def process(self, gm_callbacks):
        """gm_callbacks is the namedtuple from a GameManager
        containing methods that can be used to get different
        state representations.

        Returns a numpy array of 3 dimensions, where the last 
        dimension is time.
        """

        frame = self.get_basis_frame(gm_callbacks)
        state = self.time_concatenate(frame)
        return state

    def trace(self, gm_callbacks):
        """Returns an np.array of shape scale_shape
        which is a weighted average over the last n_frame_concat
        frames, plus the current frame
        """
        state = self.process(gm_callbacks)
        weights = [x+1 for x in reversed(range(state.shape[0]))]
        return np.average(state, axis=0, weights=weights)

    def time_max(self, frame):
        lastframes = self._unprocessed[:]
        self._unprocessed.append(frame)

        return np.max(np.stack([frame] + lastframes), axis=0)

    def luminance(self, frame):
        # Use luma constants
        # https://en.wikipedia.org/wiki/HSL_and_HSV#Lightness
        weights = np.array([0.3, 0.59, 0.11])
        weighted = np.multiply(frame, weights)
        return np.sum(weighted, axis=2)

    def resize(self, frame):
        return imresize(frame, self.scale_shape, 'bilinear')

    def get_basis_frame(self, gm_callbacks):
        """Take max for each channel over this and last frames
        Convert to luminance
        Scale down
        """
        frame = gm_callbacks.rgb()
        frame = self.time_max(frame)
        frame = self.luminance(frame)
        frame = self.resize(frame)

        return frame

    def time_concatenate(self, frame):
        frames = [frame] + self._processed[:]
        frames = np.stack(frames, axis=0)
        self._processed.append(frame)
        return frames

    def get_settings(self):
        return {
            'scale_shape': self.scale_shape,
            'n_frame_concat': self._processed.capacity(),
            'n_frame_max': self._unprocessed.capacity(),
        }


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
class DummyCNN(NN):
    def __init__(self, n_outputs):
        self.n_outputs = n_outputs

    def train(self, memory):
        pass

    def predict(self, state):
        return randrange(self.n_outputs)
        


class CNN(NN):
    """
    Convolutional Neural Networks! Yay
    """
    def __init__(
            self, n_inputs=(64, 64), n_outputs=12, n_channels=4, 
            config='deepmind', gamma=0.99, target_copy_interval=10,
            batch_size=32):
        super(CNN, self).__init__(name='CNN', version='1')

        self.gamma = gamma
        self.target_copy_interval = target_copy_interval
        self.iteration = 0
        self.batch_size = batch_size

        configurations = {
            'deepmind': self._build_deepmind
        }

        try:
            self.network = configurations[config](n_inputs, n_outputs, n_channels)
        except KeyError:
            print "Invalid network configuration, choose one of {}".format(self.configurations.keys())
            raise

        self.network_next = self.network

        # Theano variable types for compiled theano expressions.
        # 
        # lasagne uses theano expressions internally. get_outputs(layer) for
        # example, returns a theano expression that must be evaluated to get
        # the actual values present at the output neurons. 
        # 
        # These expressions and functions (like the loss function) are
        # defined here for future use. They are strongly typed and hence the
        # parameter types must be declared explicitly. That's what is done
        # here.
        
        # s, a, r, s types for network training/evaluation when using the
        # memory
        states_t = T.tensor4('states')
        next_states_t = T.tensor4('next_states')
        reward_t = T.col('rewards')
        action_t = T.icol('actions')


        self.states_shared = theano.shared(
            np.zeros((batch_size, 1, n_inputs[0], n_inputs[1]),
                     dtype=theano.config.floatX))

        self.next_states_shared = theano.shared(
            np.zeros((batch_size, 1, n_inputs[0], n_inputs[1]),
                     dtype=theano.config.floatX))

        self.rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self.actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        
        # Network outputs
        q_vals = nom.layers.get_output(self.network, states_t)
        next_q_vals = nom.layers.get_output(self.network_next, next_states_t)

        # Create a loss expression for training, i.e., a scalar objective 
        # that should be minimised. 
        # 
        # We are using a simple squared_error function like in the paper.
        # However, the target value is defined as
        #     t = r_t + discount * argmax_a' Q'(s, a')
        # so the target value is a scalar corresponding to just one of the 
        # net work's outputs.
        target = (reward_t + self.gamma * T.max(next_q_vals, axis=1, keepdims=True))
        diff = target - q_vals[T.arange(self.batch_size),
                               action_t.reshape((-1,))].reshape((-1, 1))

        loss = 0.5* diff **2
        loss = loss.mean()

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Nom offers plenty more.
        params = nom.layers.get_all_params(self.network, trainable=True)
        updates = nom.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

        givens = {
            states_t: self.states_shared,
            next_states_t: self.next_states_shared,
            reward_t: self.rewards_shared,
            action_t: self.actions_shared
        }

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        self.train_fn = theano.function([], [loss, q_vals],
                                        updates=updates, allow_input_downcast=True,
                                        givens=givens)

        self.predict_fn = theano.function([], q_vals,
                                          allow_input_downcast=True,
                                          givens={ states_t: self.states_shared})

        # Some previous network configuration
        self.p_network = self.network


    def _build_deepmind(self, n_inputs, n_output, n_channels):
        """
        Builds the CNN used in the Google deepmind Atari paper (doi:10.1038)

        Input layer:
            64x64x4 (84x84x4 in the paper)
        1st hidden: 
            32 filters of 8x8, stride 4 Rectifier
        2nd hidden:
            64 filters of 4x4, stride 2 Rectifier
        3rd hidden:
            64 filters of 3x3, stride 1 Rectifier
        4th hidden:
            512 fully-connected rectifier units
        Output layer:
            4-18 fully connected linear units

        The first 3 hidden layers are essentially filters, with an output of
        max(0, x), where x is the maximum pixel value observed in the filter
        window.

        TODO: does the filter used in the paper in fact just return 
        max(pixels)?

        The output neurons represent the Value of performing the action mapped
        to the neuron given the state that is fed into the input layer, ie.
        the Q-value
        """

        # POW: input_var must be a parameter in the network creation, otherwise
        #      Theano cannot make the connection between the loss function
        #      parameter 'inputs' and the inputs to the network
        # input layer with (unknown_batch_size, n_channels, n_rows, n_columns)
        l_in = nom.layers.InputLayer(
            shape=(None, n_channels, n_inputs[0], n_inputs[1]))

        # first hidden layer
        l_h1 = nom.layers.Conv2DLayer(
            l_in, num_filters=32, filter_size=(8, 8), stride=(4, 4),
            nonlinearity=nom.nonlinearities.rectify,
            W=nom.init.GlorotUniform())

        # second hidden layer
        l_h2 = nom.layers.Conv2DLayer(
            l_h1, num_filters=64, filter_size=(4, 4), stride=(2, 2),
            nonlinearity=nom.nonlinearities.rectify,
            W=nom.init.GlorotUniform())

        # third hidden layer
        l_h3 = nom.layers.Conv2DLayer(
            l_h2, num_filters=64, filter_size=(3, 3), stride=(1, 1),
            nonlinearity=nom.nonlinearities.rectify,
            W=nom.init.GlorotUniform())
        
        # fourth hidden layer
        l_h4 = nom.layers.DenseLayer(
            l_h3, num_units=512, 
            nonlinearity=nom.nonlinearities.rectify)

        # output layer: linearity must be explicit
        l_out = nom.layers.DenseLayer(
            l_h4, num_units=n_output,
            nonlinearity=None)

        return l_out


    def train(self, memory):
        """
        memory is a list of state-action-reward-state' tuples
        """

        self.iteration += 1
        train_err = 0
        train_batches = 0
        start_time = time.time()

        # sample random minibatch from memory
        # Should not raise ValueError anymore, as train is only called
        # when memory has content.
        samples = memory.uniform_random_sample(32)

        # TODO: this probably isn't very efficient...?
        s, a, r, s_n = zip(*samples)
        s = np.array(s)
        a = np.array(a).reshape((32, 1))
        r = np.array(r).reshape((32, 1))
        s_n = np.array(s_n)
        # print "sample={}".format(s)
        # print "actions={}".format(a)
        # print "rewards={}".format(r)
        # print "next_sample={}".format(s_n)

        # self.predict(s)
        train_batches += 1

        self.states_shared.set_value(s)
        self.next_states_shared.set_value(s_n)
        self.actions_shared.set_value(a)
        self.rewards_shared.set_value(r)

        loss, _ = self.train_fn()

        # reset the 'target network' every target_copy_interval iterations
        if self.iteration % self.target_copy_interval == 0:
            params = nom.layers.helpers.get_all_param_values(self.network)
            nom.layers.helpers.set_all_param_values(self.network_next, params)

        # Then we print the results for this training set
        print "Training took {}s, loss: {}".format(time.time() - start_time, loss)

        return np.sqrt(loss)


    def predict(self, state):
        # TODO: we want to get the index corresponding to the maximum
        #       value of the network outputs (ie, argmax_a Q(s, a))
        #       get_output returns a theano expression that is evaluated
        #       using eval(). See here:
        #       https://github.com/Lasagne/Lasagne/issues/475

        states = np.zeros((32, 4, 64, 64), dtype=theano.config.floatX)
        states[0, ...] = state
        self.states_shared.set_value(states)
        output = self.predict_fn()[0]
        print "output {}".format(output)
        return np.argmax(output)

    def _get_layer_settings(self, layer):
        settings = {
            'name': layer.name,
            'type': str(type(layer)),
            'input_shape': getattr(layer, 'input_shape' , None),
            'output_shape': getattr(layer, 'output_shape' , None),
            'nonlinearity': layer.nonlinearity.func_name if 'nonlinearity' in dir(layer) else None,
            # Convolutional layers
            'stride': getattr(layer, 'stride', None),
            'num_filters': getattr(layer, 'num_filters', None),
            'filter_size': getattr(layer, 'filter_size', None),
            # Dense layers
            'num_units': getattr(layer, 'num_units' , None),
        }
        
        # Remove Nones
        for k,v in list(settings.iteritems()):
            if v is None:
                del settings[k]

        return settings

    def get_settings(self):
        layers = nom.layers.get_all_layers(self.network)
        network = map(self._get_layer_settings, layers)
        
        settings = {
            "gamma": self.gamma,
            "target_copy_interval": self.target_copy_interval,
            "network": network,
        }

        settings.update(super(CNN, self).get_settings())

        return settings

