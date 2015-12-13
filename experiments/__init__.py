import importlib
import os

def list():
    """Returns a list of available experiment names
    """

    dircontent = os.listdir('experiments')
    pyfiles = filter(lambda x: x.endswith('.py'), dircontent)
    experiments = filter(lambda x: not '__init__' in x, pyfiles)
    names = map(lambda x: x[:-3], experiments)
    return names

def run(experiment):
    """experiment is the name of one of the
    experiments contained in this package.

    This commands runs the given experiment.
    """
    module = 'experiments.' + experiment
    return importlib.import_module(module)

def get_results_path(experiment):
    if experiment.startswith('experiments.'):
        experiment = experiment[12:]

    if has(experiment):
        return os.path.join('results', experiment)
    return None

def get_newest_results_path(experiment):
    res_path = get_results_path(experiment)
    if not os.path.exists(res_path):
        return None
    
    dircontent = os.listdir(res_path)
    if len(dircontent) == 0:
        return None
    newest = sorted(dircontent)[-1]
    return os.path.join(res_path, newest)

def has(experiment):
    available = list()

    if experiment in available:
        return experiment

    print "No experiment known under the name '{}'".format(experiment)
    print "Available experiments are"
    for v in sorted(available):
        print "\t{}".format(v)
    return False
