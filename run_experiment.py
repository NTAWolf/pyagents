#!/usr/bin/env python


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Experiment handler for pyagents using ALE.')
    parser.add_argument('experiment', help=('The name of the experiment you want to run. '
                                            'If it is NAME, there should be a file in '
                                            'experiments/NAME.py that runs the actual '
                                            'experiment. Additionally, this file should be '
                                            'modified to contain the necessary information '
                                            'about that experiment.'))
    
    experiment = parser.parse_args().experiment

    if experiment == 'testbed':
        from experiments import testbed


if __name__ == '__main__':
    main()
else:
    raise ImportError("The run_experiment module can only be run as main")