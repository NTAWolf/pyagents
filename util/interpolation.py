def linear_latch(x, x1, x2, y1, y2):
    """Assuming x1 < x2, interpolates y1 to y2 linearly
    over the range x1 to x2, with respect to the current
    value x. When x is outside the range [x1,x2], the levels
    from y1 and y2 are held.
    """
    if x < x1:
        return y1
    if x > x2:
        return y2
    return y1 + (y2 - y1) * ((x - x1) / (x2 - x1))


class LinearLatch(object):
    """Object encapsulating linear_latch method"""

    def __init__(self, x1, x2, y1, y2):
        self.args = (x1, x2, y1, y2)

    def interpolate(self, x):
        return linear_latch(x, *self.args)
