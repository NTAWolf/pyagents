import matplotlib
matplotlib.use('Agg') # us a backend that doesn't require a DISPLAY

import matplotlib.pyplot as plt
import os
from util.logging import CSVLogger

class DataCapture(object):
    def __init__(self, path, fprefix='frame_', tprefix='trace_'):
        self.path = path
        self.fprefix = fprefix
        self.tprefix = tprefix
        self.img = None

    def save_frame(self, data, filename):
        path = os.path.join(self.path, self.fprefix + filename + '.png')
        if self.img is None:
            self.img = plt.imshow(data)
        else:
            self.img.set_data(data)
        plt.savefig(path)
        print "Saved figure in {}".format(path)

    def write_trace(self, header, data, filename):
        c = CSVLogger(self.path + '/' + self.tprefix + filename + '.csv', header)
        for d in data:
            c.write(*d)
