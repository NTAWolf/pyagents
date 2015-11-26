"""This module defines utility methods
that perform operations on lists or 
listlike objects.
"""


def product(*args):
    res = 1
    for arg in args:
        res *= arg
    return res


def sublists(listlike, maxlen=None):
    """Given [a b c d]
        yields
            [a]
            [a b]
            [a b c]
            [a b c d]
    """
    for i in range(len(listlike)):
        yield listlike[:i]


def listhash(listlike):
    """list hash
    """
    if len(listlike) == 0:
        return hash(None)
    res = hash(listlike[0])
    for v in listlike[1:]:
        res = res ^ hash(v)
    return res
        