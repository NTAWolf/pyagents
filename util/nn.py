import numpy as np
import theano.tensor as T
import theano
import lasagne as nom
import time

from random import randrange

from scipy.misc import imresize

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
        """Called by the GameManager when it is
        time to store this object's settings

        Returns a dict representing the settings needed to 
        reproduce this object.
        """
        return dict([
            ('scale_shape', self.scale_shape),
            ('n_frame_concat', self._processed.capacity()),
            ('n_frame_max', self._unprocessed.capacity()),
        ])


class NN(object):
    """
    Base class for neural network providing a uniform interface to build, 
    train and evaluate various networks.
    """
    def __init__(self, name, version):
        pass

    def train(self, td):
        raise NotImplementedError()

    def predict(self, state):
        raise NotImplementedError()


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
            self, n_inputs=(64, 64), n_output=16, n_channels=4, config='deepmind', gamma=0.99,
            target_copy_interval=10):
        super(CNN, self).__init__(name='CNN', version='1')
        try:
            self.network, self.train_fn = self.configurations[config](n_inputs, n_output, n_channels)
        except KeyError:
            print "Invalid network configuration, choose one of {}".format(self.configurations.keys())
            raise

        # Some previous network configuration
        self.p_network = self.network
        self.gamma = gamma
        self.target_copy_interval = target_copy_interval
        self.iteration = 0


    def _make_targets(self, reward, next_state):
        # target value y = r_j                                 if episode terminates at time j+1
        #                  r_j + g*max_a' Q'(s_{t+1}, a'; W')  otherwise 
        # TODO: how do I detect a terminal state?
        if not next_state:
            return reward
        else:
            output = lasagna.layers.get_output(self.p_network, next_state)
            return reward + self.gamma * output.max()
        

    def train(self, memory):
        """
        experience is a list of state-action-reward-state' tuples
        """

        self.iteration += 1
        train_err = 0
        train_batches = 0
        start_time = time.time()

        # sample random minibatch from experience
        try:
            samples = memory.uniform_random_sample(32)
        except ValueError:
            # TODO: that's a bit hacky
            return

        for mem in samples:
            s, a, r, s_n = mem
            target = self._make_target(r, s_n)
            train_err += self.train_fn(s, target, a)
            train_batches += 1

        # reset the 'target network' every target_copy_interval iterations
        if self.iteration % self.target_copy_interval == 0:
            params = nom.layers.helpers.get_all_param_values(self.network)
            nom.layers.helpers.set_all_param_values(self.p_network, params)

        # Then we print the results for this training set
        print "Training took {:.3f}s, loss: {:.6f}".format(time.time() - start_time, train_err / train_batches)


    def predict(self, state):
        # TODO: we want to get the index corresponding to the maximum
        #       value of the network outputs (ie, argmax_a Q(s, a))
        #       get_output returns a theano expression that is evaluated
        #       using eval(). See here:
        #       https://github.com/Lasagne/Lasagne/issues/475
        outputs = nom.layers.get_output(self.network)
        print "outputs {}".format(outputs.eval())
        return 0


    def _build_deepmind(n_inputs, n_output, n_channels):
        """
        Builds the CNN used in the Google deepmind Atari paper (doi:10.1038)

        Input layer:
            64x64x4
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
        input_var = T.tensor4('input')   # dimensions: num_batch * num_features
        
        # input layer with (unknown_batch_size, n_channels, n_rows, n_columns)
        l_in = nom.layers.InputLayer(
            shape=(None, n_channels, n_inputs[0], n_inputs[1]),
            input_var=input_var)

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

        # output layer
        l_out = nom.layers.DenseLayer(
            l_h4, num_units=n_output)


        # Prepare Theano variables for inputs and targets
        action_var = T.iscalar('action')  # dimensions: num_batch
        target_var = T.ivector('target')  # dimensions: num_batch
        output_var = nom.layers.get_output(l_out)

        # Create a loss expression for training, i.e., a scalar objective that should
        # be minimised. We are using a simple squared_error function like in the paper
        # TODO: updates are done using the SE of just the action that was performed
        #       in this given sample. Therefore, the loss function should just
        #       calculate the SE(y - Q(s, a)) for a particular a, not all a...
        prediction = nom.layers.get_output(l_out)
        loss = (prediction[action_var] - target_var)**2
        loss = loss.mean()

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Nom offers plenty more.
        params = nom.layers.get_all_params(l_out, trainable=True)
        updates = nom.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var, action_var], loss, updates=updates)

        return l_out, train_fn


    configurations = {
        'deepmind': _build_deepmind
    }

