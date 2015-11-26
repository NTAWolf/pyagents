#!/usr/bin/env python

import argparse
import os


EXPERIMENTS_PACKAGE = "experiments"


def load_experiment(name):
    name = EXPERIMENTS_PACKAGE + "." + name
    return __import__(name, fromlist=[''])


def main():
    parser = argparse.ArgumentParser(
        description='Experiment handler for pyagents using ALE.')
    parser.add_argument('experiment',
                        help=('The name of the experiment you want to run. '
                              'If it is NAME, there should be a file in '
                              'experiments/NAME.py that runs the actual '
                              'experiment.')
                        )

    available = os.listdir(EXPERIMENTS_PACKAGE)
    
    # only .py files (not __init__), and remove the .py
    available = [d[:-3]
                 for d in available if d.endswith('.py') and (not '__init__' in d)]

    experiment = parser.parse_args().experiment

    if experiment in available:
        load_experiment(experiment)
    else:
        print "No experiment known under the name '{}'".format(experiment)
        print "Available experiments are"
        for v in sorted(available):
            print "\t{}".format(v)

if __name__ == '__main__':
    main()
else:
    raise ImportError("The run_experiment module can only be run as main")
