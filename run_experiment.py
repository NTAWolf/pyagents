#!/usr/bin/env python


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Experiment handler for pyagents using ALE.')
    parser.add_argument('experiment', 
                        help=('The name of the experiment you want to run. '
                              'If it is NAME, there should be a file in '
                              'experiments/NAME.py that runs the actual '
                              'experiment. Additionally, this file should be '
                              'modified to contain the necessary information '
                              'about that experiment.'))

    available = ("testbed",
                 "testbed2")

    experiment = parser.parse_args().experiment

    if experiment == 'testbed':
        from experiments import testbed
    elif experiment == 'testbed2':
        from experiments import testbed2
    else:
        print "No experiment known under the name '{}'".format(experiment)
        print "Available experiments are"
        for v in sorted(available):
            print "\t{}".format(v)


if __name__ == '__main__':
    main()
else:
    raise ImportError("The run_experiment module can only be run as main")
