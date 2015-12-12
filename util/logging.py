from __future__ import print_function


class CSVLogger(object):

    def __init__(self, path, header=None, print_items=False):
        self.path = path
        self.clear(header)
        self.print_items = print_items

    def write(self, *args):
        args = ','.join([str(a) for a in args])
        if self.print_items:
            print(args)
        with open(self.path, 'a') as f:
            f.write(args + '\n')

    def clear(self, newitem=None):
        with open(self.path, 'w') as f:
            if newitem:
                f.write(newitem + '\n')

class Logger(object):

    def __init__(self, levels, print_threshold, path, prepend=None, formatter=None):
        self.levels = levels
        self.print_threshold = print_threshold
        self.prepend = (str(prepend) + ': ') if prepend else ''
        self.formatter = formatter or (
            lambda x, y: "{}{}: {}".format(self.prepend, x, y))
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