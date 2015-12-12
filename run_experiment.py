#!/usr/bin/env python

import argparse

import experiments

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
    if experiments.has(experiment):
        experiments.run(experiment)
    
if __name__ == '__main__':
    main()
else:
    raise ImportError("The run_experiment module can only be run as main")
