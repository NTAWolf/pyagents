from random import randrange


class RepeatManager(object):
    """Handles the overhead of repeating
    a value a number of times.

    Usage example:
        val = rm.next() # Get repeated va
        if val:
            return val
        # else continue with method, generating a new value
        rm.set(val)
    """

    def __init__(self, n_repetitions):
        """n_repetitions is the number of times
        RepeatManager.next shall return a value set with
        RepeatManager.set, before returning None.
        """
        self.n_repetitions = n_repetitions
        self.n = 0
        self.value = None

    def next(self):
        """Returns repeated value or None.
        If None, a new value should be set using .set()
        Increments the repetition counter.
        """
        if self.value == None or self.n >= self.n_repetitions:
            return None
        self.n += 1
        return self.value

    def set(self, value):
        """Set a new value to be repeated the
        previously specified number of times
        """
        self.value = value
        self.n = 0



class LinearInterpolationManager(object):
    """Handles linear inter- and extrapolation w.r.t.
    a list of 2D points (X,Y), where the X values
    are assumed to be counts. The internal count is 
    incremented every time .next is called.

    Extrapolation is done by keeping the nearest given Y-value.
    """

    def __init__(self, handles):
        
        self.handles = handles
        self.c = 0

    def next(self):
        # Count of calls greater than final x-value?
        if self.c > self.handles[-1][0]:
            return self.handles[-1][1]

        self.c += 1

        # Count of calls less than first x-value?
        if self.c < self.handles[0][0]:
            return self.handles[0][1]

        for i in range(len(self.handles) - 1):
            if self.c > self.handles[i][0]:
                break
        return self.linear_interpolation(i, i+1)

    def linear_interpolation(self, hi1, hi2):
        x1, y1 = self.handles[hi1]
        x2, y2 = self.handles[hi2]

        return y1 + (y2 - y1) * ((self.c - x1) / (x2 - x1))

    def get_settings(self):
        """Called by the GameManager when it is
        time to store this object's settings

        Returns a dict representing the settings needed to 
        reproduce this object.
        """
        return {"handles": self.handles}
