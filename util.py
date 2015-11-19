
from __future__ import print_function
import os

class Logger(object):
    def __init__(self, levels, print_threshold, path, prepend='', formatter=None):
        self.levels = levels
        self.print_threshold = print_threshold
        self.prepend = prepend
        self.formatter = formatter or (lambda x,y: "{}:{}: {}".format(self.prepend,x,y))
        self.path = path
        self.clear()
        self._initiate_levels()

    def write(self, msg):
        with open(self.path, 'a') as f:
            f.write(msg + '\n')

    def _initiate_levels(self):
        import types
        for level in self.levels:
            mt = types.MethodType(self._make_level(level), self, Logger)
            setattr(self, level, mt)

    def _make_level(self, level):
        def f(self, message):
            msg = self.formatter(level, message)
            if self.levels.index(level) >= self.levels.index(self.print_threshold):
                print(msg)
            self.write(msg)
        f.__name__ = level
        return f

    def clear(self):
        with open(self.path, 'w') as f:
            pass
