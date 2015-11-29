import numpy as np

class CircularList(object):
    """A circular list that stores items in order.
    It is intended for use in experience replay.

    It support uniform random sampling, usual indexing, and slice notation.
    """

    def __init__(self, length):
        self.data = [None for _ in xrange(length)]
        self.i = -1  # Index of the last changed object
        # Flag set to True when the whole list is filled out and we start
        # looping
        self.full = False

    def append(self, value):
        self.i += 1
        if self.i >= len(self.data):
            self.i = 0
            self.full = True
        self.data[self.i] = value

    def insert(self, value):
        print "CircularList.insert is deprecated!"
        self.append(value)

    def batch_insert(self, values):
        for v in values:
            self.insert(v)

    def uniform_random_sample(self, sample_size):
        if self.full:
            data = self.data
        else:
            data = self.data[:self.i + 1]
        return np.random.choice(data, sample_size, replace=True)  # uniform

    def _is_valid_index(self, index):
        return index >= 0 and ((index < len(self.data)) if self.full else (index <= self.i))

    def _get_single_item(self, index):
        if self.i == -1:
            raise IndexError("No elements in CircularList yet.")
        if not self._is_valid_index(index):
            raise IndexError("Invalid index {}. Should be in range (0, {})".format(
                index, (len(self.data) if self.full else self.i)))

        i = self.i - index
        if self.full:
            i = i % len(self.data)

        return self.data[i]

    def __getitem__(self, key):
        """index 0 is the most recent inserted object.
        index length-1 is the oldest object in memory.

        Supports slice notation
        """
        if type(key) == slice:
            start = key.start or 0
            stop = key.stop or (
                len(self.data) if self.full else self.i + 1)
            step = key.step or 1

            return [self[i] for i in xrange(start, stop, step)]
        else:
            return self._get_single_item(key)

    def __len__(self):
        if self.full:
            return len(self.data)
        return self.i + 1

    def __repr__(self):
        return str(self[:])

    def capacity(self):
        return len(self.data)
