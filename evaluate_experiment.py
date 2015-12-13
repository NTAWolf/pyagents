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

class Evaluator(object):

    # 1000 forms of initialization
    def __init__(s, results_dir):
        s.results_dir = results_dir
        s.init_stats()
        s.init_settings()
        s.init_eval_path()

        s.evaluate()

    def init_stats(s):
        stats_path = os.path.join(s.results_dir, 'stats.log')
        stats = pd.read_csv(stats_path)
        s.stats = stats

    def init_settings(s):
        settings_path = os.path.join(s.results_dir, 'settings')

        with open(settings_path, 'r') as f:
            settings = json.load(f)

        s.settings = settings

    def init_eval_path(s):
        eval_path = os.path.join(s.results_dir,'evaluation')
        if os.path.exists(eval_path):
            shutil.rmtree(eval_path)
        os.makedirs(eval_path)
        s.eval_path = eval_path

    # Actual evaluation
    def evaluate(s):
        s.plot_mean_reward_per_episode()

    def plot_mean_reward_per_episode(s):
        episode_mean = s.stats.groupby('episode').mean()
        episode_mean.total_reward.plot(style=['o'])
        plt.ylabel('Mean total reward over {} epochs'.format(
                        len(s.stats.epoch.unique())))
        plt.title('{}'.format(s.settings['game_name']))

        s.expand_plot_lims()
        s.savefig('mean_reward_per_episode.png')

    # Plot utils
    def expand_plot_lims(s, d=1):
        lim = plt.xlim()
        plt.xlim((lim[0]-d,lim[1]+d))
        lim = plt.ylim()
        plt.ylim((lim[0]-d,lim[1]+d))

    def savefig(s, filename):
        fig_path = os.path.join(s.eval_path, filename)
        plt.savefig(fig_path)
        print("Saved fig in {}".format(fig_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experiment evaluator for pyagents.')
    parser.add_argument('experiment',
                        help=('Name of experiment to evaluate. Automatically '
                              'uses the results of the very latest run of '
                              'the experiment.'),
                        default='HELP_ME_I_FORGOT_TO_GIVE_IT_AN_EXPERIMENT',
                        nargs='?'
                        )

    experiment = parser.parse_args().experiment
    if experiments.has(experiment):
        results_dir = experiments.get_newest_results_path(experiment)
        Evaluator(results_dir)
        # evaluate(results_dir)
else:
    raise ImportError("The evaluate_experiment module can only be run as main")
