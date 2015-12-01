#!/usr/bin/env python

import argparse
import importlib
import os


EXPERIMENTS_PACKAGE = "experiments"


def load_experiment(name):
    name = EXPERIMENTS_PACKAGE + "." + name
    return importlib.import_module(name)


def get_available_experiments():
    dircontent = os.listdir(EXPERIMENTS_PACKAGE)
    pyfiles = filter(lambda x: x.endswith('.py'), dircontent)
    experiments = filter(lambda x: not '__init__' in x, pyfiles)
    names = map(lambda x: x[:-3], experiments)
    return names


def main():
    parser = argparse.ArgumentParser(
        description='Experiment handler for pyagents using ALE.')
    parser.add_argument('experiment',
                        help=('The name of the experiment you want to run. '
                              'If it is NAME, there should be a file in '
                              'experiments/NAME.py that runs the actual '
                              'experiment.'),
                        default='HELP_ME_I_FORGOT_TO_GIVE_IT_AN_EXPERIMENT',
                        nargs='?'
                        )

    experiment = parser.parse_args().experiment
    available = get_available_experiments()

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
