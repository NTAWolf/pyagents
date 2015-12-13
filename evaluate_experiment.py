#!/usr/bin/env python

from __future__ import print_function
import argparse

import os
import shutil
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn # Makes plots look prettier

import experiments

def evaluate(results_dir):
    print("Evaluating experiment from directory {}".format(results_dir))

    stats_path = os.path.join(results_dir, 'stats.log')
    settings_path = os.path.join(results_dir, 'settings')

    with open(settings_path, 'r') as f:
        settings = json.load(f)


    stats = pd.read_csv(stats_path)
    episode_mean = stats.groupby('episode').mean()
    episode_mean.total_reward.plot(style=['o'])
    plt.ylabel('Mean total reward over {} epochs'.format(
                    len(stats.epoch.unique())))
    plt.title('{}'.format(settings['game_name']))

    lim = plt.xlim()
    plt.xlim((lim[0]-1,lim[1]+1))
    lim = plt.ylim()
    plt.ylim((lim[0]-1,lim[1]+1))

    eval_path = os.path.join(results_dir,'evaluation')
    if os.path.exists(eval_path):
        shutil.rmtree(eval_path)
    os.makedirs(eval_path)

    fig_path = os.path.join(eval_path, 'mean_reward_per_episode.png')
    plt.savefig(fig_path)

    print("Saved fig in {}".format(fig_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experiment evaluator for pyagents.')
    parser.add_argument('experiment',
                        help=('Name of experiment to evaluate. Automatically '
                              'uses the results of the very latest run of '
                              'the experiment.'))

    experiment = parser.parse_args().experiment
    if experiments.has(experiment):
        results_dir = experiments.get_newest_results_path(experiment)
        evaluate(results_dir)
else:
    raise ImportError("The evaluate_experiment module can only be run as main")
